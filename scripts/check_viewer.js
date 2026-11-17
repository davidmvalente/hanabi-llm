const { chromium } = require('playwright');
// usage: node scripts/check_viewer.js <screenshot-dir> [url]   (needs Node and the  package)
const SCR = process.argv[2];
const URL = process.argv[3] || 'http://127.0.0.1:8791/';
const assert = (cond, msg) => { if (!cond) { throw new Error('ASSERTION FAILED: ' + msg); } console.log('  ok  ' + msg); };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1000, height: 1250 }, deviceScaleFactor: 2 });
  const errors = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(URL);
  await page.waitForSelector('#status .done', { timeout: 20000 });          // the stream ends with `done`

  const n = await page.evaluate(() => frames.length);
  console.log(`frames received: ${n}`);
  assert(n > 60, 'the whole feed arrived');

  // 1. default: see everything
  assert(await page.locator('#hands .tile.back').count() === 0, 'default view: no hand is blinded');
  assert(await page.locator('#hands .hand').count() === 3, 'three hands are shown');
  const inState = await page.evaluate(() => frames[frames.length - 1].state.hands.flat().length);
  assert(await page.locator('#hands .card').count() === inState, `every card of the final state is shown face-up (${inState}: hands shrink once the deck is empty)`);
  assert(await page.locator('#hints .pip').count() === 8 && await page.locator('#fuses .pip').count() === 3, '8 hint tokens and 3 fuse tokens are drawn');
  assert(await page.locator('#stacks .stack').count() === 5, 'five stacks');
  const nonDraw = await page.evaluate((upto) => frames.slice(1, upto + 1).filter((f) => f.record.kind !== 'draw').length, n - 1);
  assert((await page.locator('#log div').count()) === nonDraw, `the log has one line per action (${nonDraw}); a draw joins the line of the action it followed`);
  const texts = await page.locator('#log div').allInnerTexts();
  assert(texts.every((t) => !/^T\d+ \w+ drew/.test(t)), 'no line is a bare draw');
  assert(texts.some((t) => /^T\d+ \w+ hinted \w+: positions? [\d, and]+ (is|are) (red|yellow|green|blue|white|rank \d)$/.test(t)), 'hint lines read "Alice hinted Bob: position 3 is green"');
  assert(texts.some((t) => /^T\d+ \w+ (played|discarded) (red|yellow|green|blue|white) \d from position \d( ✓ placed| ✗ misfire \(fuse lost\))?( and drew (red|yellow|green|blue|white) \d)?$/.test(t)), 'cards read "blue 1", never a bare number');
  assert((await page.locator('#log .chip').allInnerTexts()).every((t) => /^(red|yellow|green|blue|white) \d$/.test(t)), 'every card chip carries its colour in its text');
  assert(texts.every((t) => /^T\d+ /.test(t) || t.startsWith('end')), 'every line starts "T7 " with a space after the turn');
  assert(await page.locator('#log .pname').first().evaluate((e) => e.style.color !== ''), 'player names are coloured');
  assert(await page.locator('#status').innerText().then((t) => t.includes('finished')), 'status says finished');

  // 2. scrub to mid-game and screenshot, default view
  const mid = 38;
  await page.$eval('#scrub', (el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }, mid);
  assert(await page.locator('#pos').innerText() === `${mid}/${n - 1}`, 'scrubbing moves to frame ' + mid);
  const nonDrawMid = await page.evaluate((m) => frames.slice(1, m + 1).filter((f) => f.record.kind !== 'draw').length, mid);
  assert((await page.locator('#log div').count()) === nonDrawMid, 'the log shows only events up to that frame');
  assert(await page.locator('#follow').isChecked() === false, 'scrubbing back stops following live');
  await page.screenshot({ path: `${SCR}/shot_default.png`, fullPage: true });

  // 3. blind one player: their cards turn face-down, the others stay visible
  await page.selectOption('#blind', 'Bob');
  const bobRow = page.locator('#hands .hand').nth(1);
  const bobN = await page.evaluate((m) => frames[m].state.hands[1].length, mid);
  assert((await bobRow.locator('.tile.back').count()) === bobN, `all ${bobN} of Bob's cards are face-down`);
  assert((await page.locator('#hands .tile.back').count()) === bobN, 'nobody else is blinded');
  assert((await bobRow.innerText()).includes('blinded'), 'the blinded row says so');
  await page.screenshot({ path: `${SCR}/shot_blind.png`, fullPage: true });

  // 4. back to the live end, blind Alice, mobile width
  await page.click('#last');
  assert(await page.locator('#follow').isChecked(), 'jumping to the latest resumes following');
  await page.selectOption('#blind', 'Alice');
  await page.setViewportSize({ width: 390, height: 1500 });
  await page.screenshot({ path: `${SCR}/shot_mobile_end.png`, fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  assert(!overflow, 'no horizontal scroll at 390px');

  assert(errors.length === 0, 'no console or page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  await browser.close();
})().catch((e) => { console.error(e.message); process.exit(1); });
