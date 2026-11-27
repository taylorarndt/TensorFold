// With playwright-core installed, run against a local TensorFold Web build:
// node tests/workspace-focus.cjs http://127.0.0.1:8090
const assert = require('node:assert/strict')
const { chromium } = require('playwright-core')

async function checkFocus(base) {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_EXECUTABLE_PATH })
  try {
    const page = await browser.newPage()
    const model = {
      id: 'example/focus', name: 'Focus model', vendor: 'Example', family: 'Example',
      description: 'Keyboard test', installed: true, runs_here: true, tags: ['chat'], backends: [],
    }
    await page.route('**/api/catalog', route => route.fulfill({ json: { models: [model] } }))
    await page.goto(new URL('/#models', base).href)
    const opener = page.getByRole('link', { name: 'Compute and clusters' })
    const close = page.getByRole('button', { name: 'Close compute and clusters' })
    const focused = locator => locator.evaluate(el => el === document.activeElement)
    for (const [key, title] of [['Enter', 'Models'], ['4', 'Learning'], ['5', 'API']]) {
      await opener.press('Enter')
      await close.waitFor()
      if (key === 'Enter') await close.press('Enter')
      else await page.keyboard.press(key)
      await close.waitFor({ state: 'detached' })
      assert.equal(await page.title(), title + ' · TensorFold Web')
      assert.ok(await focused(key === 'Enter' ? opener : page.getByRole('main')), title)
    }
    await page.getByRole('link', { name: 'My models' }).click()
    const row = page.getByRole('button', { name: model.name, exact: true })
    const closeDetails = page.getByRole('button', { name: 'Close model details' })
    const search = page.getByRole('searchbox', { name: 'Search models' })
    for (const hideRow of [false, true]) {
      await row.press('Enter')
      await closeDetails.waitFor()
      if (hideRow) {
        await search.fill('no match')
        await row.waitFor({ state: 'detached' })
      }
      await closeDetails.press('Enter')
      await closeDetails.waitFor({ state: 'detached' })
      assert.ok(await focused(hideRow ? search : row), 'Closing model details')
    }
  } finally {
    await browser.close()
  }
}

checkFocus(process.argv[2]).then(() => console.log('Workspace focus checks passed'))
  .catch(error => { console.error(error); process.exitCode = 1 })
