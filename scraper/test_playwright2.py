import asyncio
from playwright.async_api import async_playwright

async def wait_for_options(page, select_id):
    # wait until the select has more than 1 option, or some timeout
    for _ in range(20): # max 10 seconds
        await page.wait_for_timeout(500)
        options = await page.locator(f"#{select_id} option").count()
        if options > 0:
            return True
    return False

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Loading page...")
        await page.goto("https://josaa.admissions.nic.in/applicant/seatmatrix/OpeningClosingRankArchieve.aspx", wait_until="networkidle")
        
        await page.locator("#ctl00_ContentPlaceHolder1_ddlYear").select_option(label="2023", force=True)
        print("Selected 2023, waiting for round...")
        await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlroundno")
        
        opts = await page.locator("#ctl00_ContentPlaceHolder1_ddlroundno option").all_inner_texts()
        print("Rounds:", opts)
        
        await page.locator("#ctl00_ContentPlaceHolder1_ddlroundno").select_option(label="6")
        print("Selected 6, waiting for Instype...")
        await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlInstype")
        
        opts = await page.locator("#ctl00_ContentPlaceHolder1_ddlInstype option").all_inner_texts()
        print("InstType:", opts)

        # we will want ALL for the rest! Let's choose ALL and see
        await page.locator("#ctl00_ContentPlaceHolder1_ddlInstype").select_option(label="ALL")
        await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlInstitute")
        
        await page.locator("#ctl00_ContentPlaceHolder1_ddlInstitute").select_option(label="ALL")
        await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlBranch")
        
        await page.locator("#ctl00_ContentPlaceHolder1_ddlBranch").select_option(label="ALL")
        await wait_for_options(page, "ctl00_ContentPlaceHolder1_ddlSeatType")
        
        await page.locator("#ctl00_ContentPlaceHolder1_ddlSeatType").select_option(label="ALL")
        print("Selected all ALLs. Now submitting...")
        
        await page.locator("#ctl00_ContentPlaceHolder1_btnSubmit").click()
        
        # wait for table
        try:
            await page.wait_for_selector("#ctl00_ContentPlaceHolder1_GridView1", timeout=60000)
            print("Table loaded!")
            # Get table HTML
            html = await page.locator("#ctl00_ContentPlaceHolder1_GridView1").inner_html()
            # We can save it or parse it
            print(f"Table HTML length: {len(html)}")
        except Exception as e:
            print("Failed to load table", e)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
