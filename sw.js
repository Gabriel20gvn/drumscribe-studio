const CACHE = 'drumscribe-shell-2026-11-22-9';
const SHELL = [
  './', './index.html', './manifest.webmanifest',
  './score-format.js',
  './drum-grid.js',
  './icons/icon-192.png', './icons/icon-512.png',
  './vendor/alphatab/alphaTab.min.js',
  './vendor/alphatab/font/Bravura.woff2',
  './vendor/alphatab/font/Bravura.woff',
  './vendor/alphatab/font/Bravura.otf',
  './vendor/alphatab/font/Bravura.eot',
  './vendor/alphatab/font/Bravura.svg'
];
const shellUrls = new Set(SHELL.map(path => new URL(path, self.registration.scope).href));

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
  event.waitUntil(Promise.all([
    caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('drumscribe-shell-') && key !== CACHE).map(key => caches.delete(key)))),
    self.clients.claim()
  ]));
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (request.mode === 'navigate') {
    event.respondWith(fetch(request).then(response => {
      if (response.ok) caches.open(CACHE).then(cache => cache.put('./index.html', response.clone()));
      return response;
    }).catch(async () => (await caches.match('./index.html')) || Response.error()));
  } else if (shellUrls.has(url.href)) {
    event.respondWith(caches.match(request).then(hit => hit || fetch(request)));
  }
});
