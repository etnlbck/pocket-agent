/**
 * Sidebar scroll-spy — highlights the active link as sections scroll into view.
 * Uses IntersectionObserver (no scroll-event listeners, no jank).
 */
(function () {
  'use strict';

  var sidebar = document.querySelector('.docs-sidebar');
  if (!sidebar) return;

  var links = Array.from(sidebar.querySelectorAll('a[href^="#"]'));
  if (!links.length) return;

  // Map href → link element for fast lookup
  var linkMap = {};
  links.forEach(function (link) {
    var id = link.getAttribute('href').slice(1);
    if (id) linkMap[id] = link;
  });

  // Get all target sections (elements with IDs that match sidebar links)
  var sections = Object.keys(linkMap)
    .map(function (id) { return document.getElementById(id); })
    .filter(Boolean);

  if (!sections.length) return;

  // Track which sections are currently visible
  var visibleSections = new Set();

  function updateActive() {
    // Find the topmost visible section
    var active = null;
    for (var i = 0; i < sections.length; i++) {
      if (visibleSections.has(sections[i].id)) {
        active = sections[i].id;
        break;
      }
    }

    // If nothing is intersecting, use scroll position to find nearest
    if (!active) {
      var scrollTop = window.scrollY + 100;
      for (var j = sections.length - 1; j >= 0; j--) {
        if (sections[j].offsetTop <= scrollTop) {
          active = sections[j].id;
          break;
        }
      }
    }

    if (!active) return;

    links.forEach(function (link) {
      link.classList.remove('active');
    });

    if (linkMap[active]) {
      linkMap[active].classList.add('active');
    }
  }

  // Observe sections entering/leaving the viewport
  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          visibleSections.add(entry.target.id);
        } else {
          visibleSections.delete(entry.target.id);
        }
      });
      updateActive();
    },
    {
      // Trigger when section header is in the top 30% of viewport
      rootMargin: '-10% 0px -70% 0px',
      threshold: 0
    }
  );

  sections.forEach(function (section) {
    observer.observe(section);
  });

  // Handle direct navigation (page load with hash, click on sidebar link)
  function onHashChange() {
    var hash = window.location.hash.slice(1);
    if (hash && linkMap[hash]) {
      links.forEach(function (link) { link.classList.remove('active'); });
      linkMap[hash].classList.add('active');
    }
  }

  window.addEventListener('hashchange', onHashChange);

  // Initial highlight on page load
  if (window.location.hash) {
    onHashChange();
  } else {
    updateActive();
  }
})();
