/* ══════════════════════════════════════════════════════
   CampusBazar — Global JavaScript
══════════════════════════════════════════════════════ */

// ── Navbar Active State ───────────────────────────────
(function() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a, .mobile-nav-links a').forEach(link => {
    const href = link.getAttribute('href');
    if (href && (path === href || (href !== '/' && path.startsWith(href)))) {
      link.classList.add('active');
    }
  });
})();

// ── Mobile Navigation ─────────────────────────────────
const hamburger = document.querySelector('.hamburger');
const mobileNav = document.querySelector('.mobile-nav');
const mobileOverlay = document.querySelector('.mobile-nav-overlay');
const mobileClose = document.querySelector('.mobile-nav-close');

function openMobileNav() {
  mobileNav?.classList.add('open');
  hamburger?.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeMobileNav() {
  mobileNav?.classList.remove('open');
  hamburger?.classList.remove('open');
  document.body.style.overflow = '';
}

hamburger?.addEventListener('click', openMobileNav);
mobileOverlay?.addEventListener('click', closeMobileNav);
mobileClose?.addEventListener('click', closeMobileNav);

// ── Toast System ─────────────────────────────────────
const toastContainer = document.querySelector('.toast-container');

function showToast(message, type = 'success', duration = 3000) {
  if (!toastContainer) {
    const tc = document.createElement('div');
    tc.className = 'toast-container';
    document.body.appendChild(tc);
  }
  const container = document.querySelector('.toast-container');
  
  const icons = { success: 'fa-check-circle', error: 'fa-times-circle', info: 'fa-info-circle' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<i class="fas ${icons[type] || icons.success}"></i><span>${message}</span>`;
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.classList.add('removing');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Modal System ─────────────────────────────────────
function openModal(id) {
  const modal = document.getElementById(id);
  if (modal) {
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (modal) {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }
}

// Close modal on overlay click
document.addEventListener('click', function(e) {
  if (e.target.classList.contains('modal-overlay')) {
    e.target.classList.remove('open');
    document.body.style.overflow = '';
  }
  if (e.target.classList.contains('modal-close')) {
    const modal = e.target.closest('.modal-overlay');
    if (modal) { modal.classList.remove('open'); document.body.style.overflow = ''; }
  }
});

// ── Wishlist ─────────────────────────────────────────
function toggleWishlist(itemId, btn) {
  fetch('/api/wishlist/toggle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item_id: itemId })
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      const isActive = data.action === 'added';
      if (btn) {
        btn.classList.toggle('active', isActive);
        const icon = btn.querySelector('i');
        if (icon) {
          icon.className = isActive ? 'fas fa-heart' : 'far fa-heart';
        }
      }
      showToast(isActive ? 'Added to wishlist!' : 'Removed from wishlist', isActive ? 'success' : 'info');
      // Update wishlist count if badge exists
      updateWishlistCount(data.wishlist.length);
    }
  })
  .catch(() => showToast('Failed to update wishlist', 'error'));
}

function updateWishlistCount(count) {
  const badge = document.querySelector('.wishlist-count');
  if (badge) {
    badge.textContent = count;
    badge.style.display = count > 0 ? 'flex' : 'none';
  }
}

// ── Delete Listing ────────────────────────────────────
function deleteListing(itemId, row) {
  if (!confirm('Are you sure you want to delete this listing? This action cannot be undone.')) return;
  fetch(`/api/listings/delete/${itemId}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        row?.remove();
        showToast('Listing deleted successfully', 'success');
        setTimeout(() => location.reload(), 1000);
      }
    })
    .catch(() => showToast('Failed to delete listing', 'error'));
}

// ── Mark as Claimed ───────────────────────────────────
function markClaimed(itemId, row) {
  if (!confirm('Mark this item as claimed/sold?')) return;
  fetch(`/api/listings/claim/${itemId}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        showToast('Listing marked as claimed!', 'success');
        setTimeout(() => location.reload(), 1000);
      }
    })
    .catch(() => showToast('Failed to update listing', 'error'));
}

// ── Animated Counter ──────────────────────────────────
function animateCounter(el, target, duration = 1500) {
  const start = 0;
  const increment = target / (duration / 16);
  let current = start;
  const timer = setInterval(() => {
    current += increment;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = Math.floor(current).toLocaleString();
  }, 16);
}

// Trigger counters when visible
const counters = document.querySelectorAll('[data-counter]');
if (counters.length > 0) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.getAttribute('data-counter'));
        animateCounter(el, target);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });
  counters.forEach(c => observer.observe(c));
}

// ── Animated SVG Rings ────────────────────────────────
function initProbRing(svgEl, percentage, color) {
  if (!svgEl) return;
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const circle = svgEl.querySelector('.ring-progress');
  if (!circle) return;
  circle.setAttribute('stroke-dasharray', circumference);
  const offset = circumference - (percentage / 100) * circumference;
  circle.style.setProperty('--target-offset', offset);
  circle.setAttribute('stroke-dashoffset', circumference);
  setTimeout(() => {
    circle.style.transition = 'stroke-dashoffset 1.2s ease-out';
    circle.setAttribute('stroke-dashoffset', offset);
  }, 200);
}

document.querySelectorAll('[data-prob-ring]').forEach(el => {
  const pct = parseInt(el.getAttribute('data-prob-ring'));
  const color = el.getAttribute('data-color') || '#556B2F';
  initProbRing(el, pct, color);
});

// ── Smooth scroll for anchor links ───────────────────
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  });
});

// ── Expose globally ───────────────────────────────────
window.toggleWishlist = toggleWishlist;
window.deleteListing = deleteListing;
window.markClaimed = markClaimed;
window.openModal = openModal;
window.closeModal = closeModal;
window.showToast = showToast;
window.initProbRing = initProbRing;
