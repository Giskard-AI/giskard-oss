/**
 * Custom functionality for Giskard documentation
 * Left sidebar scroll to current item on page load
 * Enterprise trial banner management
 */
(function() {
  'use strict';

  function scrollToCurrentItem() {
    // Find the left sidebar
    const sidebar = document.querySelector('#left-sidebar');
    if (!sidebar) return;

    // Find the current/active item with multiple selector fallbacks
    const currentItem = sidebar.querySelector('a.current') ||
                       sidebar.querySelector('.current a') ||
                       sidebar.querySelector('li.current a') ||
                       sidebar.querySelector('.toctree-l1.current a') ||
                       sidebar.querySelector('[aria-current="page"]');

    if (currentItem) {
      // Small delay to ensure DOM is fully rendered
      setTimeout(() => {
        // Get the position of the current item relative to the sidebar
        const sidebarRect = sidebar.getBoundingClientRect();
        const itemRect = currentItem.getBoundingClientRect();
        const relativeTop = itemRect.top - sidebarRect.top;
        const sidebarHeight = sidebar.clientHeight;

        // Calculate the scroll position to center the item in the sidebar
        const scrollTop = sidebar.scrollTop + relativeTop - (sidebarHeight / 2) + (itemRect.height / 2);

        // Scroll only the sidebar, not the entire page
        sidebar.scrollTo({
          top: scrollTop,
          behavior: 'smooth'
        });
      }, 100);
    }
  }

  function createEnterpriseTrialBanner() {
    // Check if banner already exists to avoid duplicates
    if (document.querySelector('.enterprise-trial-banner')) {
      return;
    }

    // Check if banner was closed in this tab session
    if (sessionStorage.getItem('enterprise-trial-banner-closed') === 'true') {
      return;
    }

    // Create banner element
    const banner = document.createElement('div');
    banner.className = 'enterprise-trial-banner';
    const baseUrl = window.location.origin;
    banner.innerHTML = `
      <span>🚀 Ready to scale your AI testing? <a href="${baseUrl}/start/enterprise-trial.html">Request your free enterprise trial today! 🛡️</a> </span>
      <button class="close-btn" aria-label="Close banner">×</button>
    `;

    // Add banner to page
    document.body.appendChild(banner);
    document.body.classList.add('banner-visible');

    // Handle close button
    const closeBtn = banner.querySelector('.close-btn');
    closeBtn.addEventListener('click', () => {
      // Hide banner with smooth animation
      banner.style.transform = 'translateY(-100%)';
      banner.style.opacity = '0';

      // Remove banner and update body class after animation
      setTimeout(() => {
        if (banner.parentNode) {
          banner.remove();
          document.body.classList.remove('banner-visible');
          // Remember that banner was closed in this tab session
          sessionStorage.setItem('enterprise-trial-banner-closed', 'true');
        }
      }, 300);
    });

    // Update banner theme when page theme changes
    function updateBannerTheme() {
      const isDark = document.documentElement.classList.contains('dark');
      banner.classList.toggle('dark', isDark);
    }

    // Initial theme check
    updateBannerTheme();

    // Watch for theme changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          updateBannerTheme();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });
  }

  // Handle Sphinx navigation events to ensure banner persists
  function handleSphinxNavigation() {
    // Small delay to ensure DOM is ready after navigation
    setTimeout(() => {
      // Only create banner if it wasn't previously closed in this tab
      if (sessionStorage.getItem('enterprise-trial-banner-closed') !== 'true') {
        createEnterpriseTrialBanner();
      }
    }, 100);
  }

  // Listen for Sphinx navigation events
  document.addEventListener('DOMContentLoaded', () => {
    // Initial banner creation
    createEnterpriseTrialBanner();

    // Initial sidebar scroll
    scrollToCurrentItem();

    // Listen for all navigation events more comprehensively
    document.addEventListener('click', (e) => {
      // Check if this is a documentation link
      if (e.target.matches('a[href*=".html"], a[href*="#"], a[href*="/"]')) {
        // Don't handle external links or anchor-only links
        const href = e.target.getAttribute('href');
        if (href && !href.startsWith('http') && !href.startsWith('mailto:') && !href.startsWith('tel:')) {
          setTimeout(handleSphinxNavigation, 300);
        }
      }
    });

    // Listen for popstate events (browser back/forward)
    window.addEventListener('popstate', () => {
      setTimeout(handleSphinxNavigation, 200);
    });

    // Listen for Sphinx's internal navigation more aggressively
    const observer = new MutationObserver((mutations) => {
      let shouldRecreateBanner = false;

      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          // Check if this looks like a page navigation
          const hasNewContent = Array.from(mutation.addedNodes).some(node =>
            node.nodeType === Node.ELEMENT_NODE &&
            (node.classList?.contains('document') ||
             node.querySelector?.('.document') ||
             node.classList?.contains('section') ||
             node.querySelector?.('.section'))
          );

          if (hasNewContent) {
            shouldRecreateBanner = true;
          }
        }
      });

      if (shouldRecreateBanner) {
        setTimeout(handleSphinxNavigation, 100);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    // Also listen for URL changes
    let currentUrl = window.location.href;
    setInterval(() => {
      if (window.location.href !== currentUrl) {
        currentUrl = window.location.href;
        setTimeout(handleSphinxNavigation, 200);
      }
    }, 100);

    // Listen for Sphinx's page load events
    document.addEventListener('DOMContentLoaded', handleSphinxNavigation, true);

    // Listen for Sphinx's navigation completion
    if (window.SphinxRtdTheme) {
      // For ReadTheDocs theme
      document.addEventListener('click', (e) => {
        if (e.target.matches('a[href*=".html"]')) {
          setTimeout(handleSphinxNavigation, 500);
        }
      });
    }
  });

  // Also handle cases where DOM is already loaded
  if (document.readyState !== 'loading') {
    createEnterpriseTrialBanner();
    scrollToCurrentItem();
  }

})();
