const CACHE_NAME = 'lumps-v1';
const urlsToCache = [
  './',
  './index.html',
  './data/current.json',
  'https://cdn.tailwindcss.com',
  'https://unpkg.com/alpinejs@3/dist/cdn.min.js',
  'https://unpkg.com/chart.js@4/dist/chart.umd.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        return response || fetch(event.request);
      }
    )
  );
});