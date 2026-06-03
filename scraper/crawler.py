import asyncio
import argparse
import os
from playwright.async_api import async_playwright

# Minimum number of table rows to consider a scrape valid.
# A valid "ALL" query typically returns thousands of rows.
# We use a dynamic threshold based on the max seen for the same year.
MIN_VALID_ROWS = 50
MAX_RETRIES = 3

async def wait_for_options(page, select_id, timeout_ms=15000):
    """Wait until the select element has more than 1 option."""
    try:
        await page.wait_for_function(
            f"document.querySelector('#{select_id}').options.length > 1",
            timeout=timeout_ms
        )
        return True
    except Exception:
        return False

async def get_options(page, select_id):
    """Get all values and labels from a select element."""
    return await page.evaluate(f"""() => {{
        const select = document.querySelector('#{select_id}');
        if (!select) return [];
        return Array.from(select.options)
            .map(o => ({{ value: o.value, label: o.innerText.trim() }}))
            .filter(o => o.value !== "0" && o.value !== "-1" && o.label !== "--- Select ---");
    }}""")

async def select_and_wait(page, select_id, value):
    """Set a hidden select's value and trigger ASP.NET postback, then wait for page reload."""
    await page.evaluate("""([id, val]) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = val;
        el.dispatchEvent(new Event('change', { bubbles: true }));
        const onchange = el.getAttribute('onchange');
        if (onchange && onchange.includes('__doPostBack')) {
            const match = onchange.match(/__doPostBack\\('([^']*)'/);
            if (match) {
                __doPostBack(match[1], '');
            }
        }
    }""", [select_id, value])
    # Wait for postback navigation to settle
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        await asyncio.sleep(3)

def count_table_rows(html):
    """Quick count of <tr> tags in the HTML."""
    return html.count("<tr") - 1

async def wait_for_stable_table(page, timeout_s=30, poll_interval_s=3):
    """Wait for the GridView table row count to stabilize.
    
    ASP.NET GridView can render rows progressively. We poll the row count
    until it stops changing, ensuring we capture the complete table.
    """
    table_id = "ctl00_ContentPlaceHolder1_GridView1"
    
    prev_count = 0
    stable_checks = 0
    elapsed = 0
    
    while elapsed < timeout_s:
        try:
            current_count = await page.evaluate(f"""() => {{
                const table = document.getElementById('{table_id}');
                if (!table) return 0;
                return table.querySelectorAll('tr').length;
            }}""")
        except Exception:
            current_count = 0
        
        if current_count > 0 and current_count == prev_count:
            stable_checks += 1
            if stable_checks >= 2:  # Stable for 2 consecutive polls
                return current_count
        else:
            stable_checks = 0
        
        prev_count = current_count
        await asyncio.sleep(poll_interval_s)
        elapsed += poll_interval_s
    
    return prev_count

def is_file_valid(filepath, expected_min_rows=MIN_VALID_ROWS):
    """Check if a previously saved HTML file has enough data rows."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r") as f:
            html = f.read()
        rows = count_table_rows(html)
        if rows < expected_min_rows:
            print(f"    ⚠ Existing file {os.path.basename(filepath)} has only {rows} rows — will re-scrape.")
            os.remove(filepath)
            return False
        return True
    except Exception:
        return False

async def scrape_single_round(page, url, year_val, year_label, round_val, round_label, filename):
    """Scrape a single round with full validation. Returns row count on success."""
    print(f"  📥 Scraping Round {round_label}...")

    # Select Round
    await select_and_wait(page, "ctl00_ContentPlaceHolder1_ddlroundno", round_val)
    await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlInstype")

    # Select ALL for cascade dropdowns
    for sel_id, next_id in [
        ("ctl00_ContentPlaceHolder1_ddlInstype", "ctl00_ContentPlaceHolder1_ddlInstitute"),
        ("ctl00_ContentPlaceHolder1_ddlInstitute", "ctl00_ContentPlaceHolder1_ddlBranch"),
        ("ctl00_ContentPlaceHolder1_ddlBranch", "ctl00_ContentPlaceHolder1_ddlSeatType"),
    ]:
        await select_and_wait(page, sel_id, "ALL")
        await wait_for_options(page, next_id)

    # SeatType — just set it
    await page.evaluate('''() => {
        const el = document.getElementById("ctl00_ContentPlaceHolder1_ddlSeatType");
        if (el) { el.value = "ALL"; }
    }''')

    # Submit
    print("    Submitting...")
    await page.locator("#ctl00_ContentPlaceHolder1_btnSubmit").click()

    # Wait for table to appear first
    await page.wait_for_selector("#ctl00_ContentPlaceHolder1_GridView1", timeout=90000)
    
    # Then wait for the row count to stabilize (critical for complete data)
    stable_rows = await wait_for_stable_table(page, timeout_s=60, poll_interval_s=3)
    print(f"    Table stabilized at {stable_rows} rows")

    # Extract HTML
    html = await page.locator("#ctl00_ContentPlaceHolder1_GridView1").evaluate("el => el.outerHTML")

    # Validate row count
    rows = count_table_rows(html)
    if rows < MIN_VALID_ROWS:
        raise ValueError(f"Only {rows} rows retrieved — likely incomplete data.")

    # Save
    with open(filename, "w") as f:
        f.write(html)

    print(f"    ✅ Saved {os.path.basename(filename)} ({rows} rows, {len(html)//1024}KB)")
    return rows

async def reset_to_year(page, url, year_val, year_label):
    """Navigate fresh and select the year. Used after errors."""
    print(f"    🔄 Resetting page and re-selecting year {year_label}...")
    await page.goto(url, wait_until="networkidle")
    await select_and_wait(page, "ctl00_ContentPlaceHolder1_ddlYear", year_val)
    await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlroundno")

async def scrape_josaa():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        url = "https://josaa.admissions.nic.in/applicant/seatmatrix/OpeningClosingRankArchieve.aspx"
        print(f"🌐 Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")

        # Output directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.join(base_dir, "raw_data")
        os.makedirs(raw_data_dir, exist_ok=True)

        # Discover all available years (oldest first)
        years = await get_options(page, "ctl00_ContentPlaceHolder1_ddlYear")
        years.reverse()
        print(f"📅 Found {len(years)} years: {[y['label'] for y in years]}")

        stats = {"success": 0, "skipped": 0, "failed": 0}
        year_max_rows = {}  # Track max rows per year for validation

        for year_opt in years:
            year_val = year_opt["value"]
            year_label = year_opt["label"]
            print(f"\n{'='*60}")
            print(f">>> YEAR {year_label}")
            print(f"{'='*60}")

            # Select Year — wrapped in try/except for resilience
            try:
                await select_and_wait(page, "ctl00_ContentPlaceHolder1_ddlYear", year_val)
                await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlroundno")
            except Exception as e:
                print(f"  ❌ Failed to select year {year_label}: {e}")
                try:
                    await reset_to_year(page, url, year_val, year_label)
                except Exception:
                    print(f"  🚫 Cannot recover year {year_label}, skipping entirely")
                    continue

            # Discover rounds
            try:
                rounds = await get_options(page, "ctl00_ContentPlaceHolder1_ddlroundno")
            except Exception as e:
                print(f"  ❌ Failed to get rounds for {year_label}: {e}")
                try:
                    await reset_to_year(page, url, year_val, year_label)
                    rounds = await get_options(page, "ctl00_ContentPlaceHolder1_ddlroundno")
                except Exception:
                    print(f"  🚫 Cannot recover, skipping year {year_label}")
                    continue
            
            print(f"Rounds: {[r['label'] for r in rounds]}")

            for round_opt in rounds:
                round_val = round_opt["value"]
                round_label = round_opt["label"]
                filename = os.path.join(raw_data_dir, f"{year_label}_round_{round_label}.html")

                # Skip valid existing files
                if is_file_valid(filename):
                    # Track row count for comparison
                    with open(filename) as f:
                        existing_rows = count_table_rows(f.read())
                    year_max_rows[year_label] = max(year_max_rows.get(year_label, 0), existing_rows)
                    print(f"  ⏭ Skipping Round {round_label} (valid, {existing_rows} rows)")
                    stats["skipped"] += 1
                    continue

                # Attempt with retries
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        row_count = await scrape_single_round(
                            page, url, year_val, year_label,
                            round_val, round_label, filename
                        )
                        year_max_rows[year_label] = max(year_max_rows.get(year_label, 0), row_count)
                        success = True
                        stats["success"] += 1
                        break
                    except Exception as e:
                        print(f"    ❌ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(3)
                            await reset_to_year(page, url, year_val, year_label)
                        else:
                            print(f"    🚫 Giving up on {year_label} Round {round_label}")
                            stats["failed"] += 1
                            await reset_to_year(page, url, year_val, year_label)

                # Polite delay
                await asyncio.sleep(2)

        await browser.close()

        # Post-scrape validation report
        print(f"\n{'='*60}")
        print(f"📊 Data Quality Report")
        print(f"{'='*60}")
        for yl in sorted(year_max_rows.keys()):
            max_r = year_max_rows[yl]
            # Check all files for this year
            year_files = sorted(glob_files(raw_data_dir, yl))
            for yf in year_files:
                with open(yf) as f:
                    rows = count_table_rows(f.read())
                pct = (rows / max_r * 100) if max_r > 0 else 0
                flag = "⚠ PARTIAL" if pct < 70 else "✅"
                print(f"  {flag} {os.path.basename(yf):30s} {rows:6d} rows ({pct:.0f}% of max)")
        
        print(f"\n{'='*60}")
        print(f"🏁 Scraping Complete!")
        print(f"   ✅ {stats['success']} saved  |  ⏭ {stats['skipped']} skipped  |  ❌ {stats['failed']} failed")
        print(f"{'='*60}")

async def scrape_current_seat_matrix():
    """Best-effort scrape of the currently available JoSAA seat matrix."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        url = "https://josaa.admissions.nic.in/applicant/seatmatrix/seatmatrixinfo.aspx"
        print(f"🌐 Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        seat_dir = os.path.join(base_dir, "seat_data")
        os.makedirs(seat_dir, exist_ok=True)

        year_text = await page.evaluate("""() => {
            const text = document.body.innerText || '';
            const match = text.match(/20\\d{2}/);
            return match ? match[0] : String(new Date().getFullYear());
        }""")

        for sel_id, next_id in [
            ("ctl00_ContentPlaceHolder1_ddlInstype", "ctl00_ContentPlaceHolder1_ddlInstitute"),
            ("ctl00_ContentPlaceHolder1_ddlInstitute", "ctl00_ContentPlaceHolder1_ddlBranch"),
            ("ctl00_ContentPlaceHolder1_ddlBranch", "ctl00_ContentPlaceHolder1_ddlSeatType"),
        ]:
            try:
                await wait_for_options(page, sel_id, timeout_ms=8000)
                await select_and_wait(page, sel_id, "ALL")
                await wait_for_options(page, next_id, timeout_ms=8000)
            except Exception:
                print(f"  Could not drive select {sel_id}; continuing best-effort.")

        try:
            await page.evaluate("""() => {
                const el = document.getElementById("ctl00_ContentPlaceHolder1_ddlSeatType");
                if (el) el.value = "ALL";
            }""")
            await page.locator("#ctl00_ContentPlaceHolder1_btnSubmit").click(timeout=15000)
            await page.wait_for_load_state("networkidle", timeout=30000)
        except Exception as exc:
            print(f"  Submit did not complete cleanly: {exc}")

        table_html = await page.evaluate("""() => {
            const tables = Array.from(document.querySelectorAll('table'));
            if (!tables.length) return '';
            tables.sort((a, b) => b.querySelectorAll('tr').length - a.querySelectorAll('tr').length);
            return tables[0].outerHTML;
        }""")

        if not table_html:
            await browser.close()
            raise RuntimeError("No seat matrix table found on the current JoSAA page.")

        html_path = os.path.join(seat_dir, f"{year_text}_seat_matrix.html")
        with open(html_path, "w") as f:
            f.write(table_html)
        print(f"✅ Saved {os.path.basename(html_path)}")

        try:
            from seats import normalize_saved_seat_html
            normalize_saved_seat_html(seat_dir)
        except Exception as exc:
            print(f"  Seat normalization failed; raw HTML is still saved: {exc}")

        await browser.close()

def glob_files(raw_data_dir, year_label):
    """Get all HTML files for a specific year."""
    import glob
    return glob.glob(os.path.join(raw_data_dir, f"{year_label}_round_*.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape JoSAA rank and optional seat matrix data")
    parser.add_argument("--kind", choices=["ranks", "seats", "all"], default="ranks")
    args = parser.parse_args()

    if args.kind == "ranks":
        asyncio.run(scrape_josaa())
    elif args.kind == "seats":
        asyncio.run(scrape_current_seat_matrix())
    else:
        asyncio.run(scrape_josaa())
        asyncio.run(scrape_current_seat_matrix())
