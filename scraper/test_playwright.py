import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await page.goto("https://josaa.admissions.nic.in/applicant/seatmatrix/OpeningClosingRankArchieve.aspx")
        
        # Give it a second to load
        await page.wait_for_timeout(2000)
        
        # Let's find all the select boxes
        selects = await page.query_selector_all("select")
        for s in selects:
            id_attr = await s.get_attribute("id")
            name_attr = await s.get_attribute("name")
            print(f"Select Found: id={id_attr}, name={name_attr}")
            
            # Print options
            options = await s.query_selector_all("option")
            print([await opt.text_content() for opt in options])

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
