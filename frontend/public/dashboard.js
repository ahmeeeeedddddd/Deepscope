// ==========================================
// DASHBOARD INTERACTIONS & ANIMATIONS
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeAnimations();
    initializeTreatmentPlan();
    initializeVisitCards();
    initializeWaveformAnimation();
});

// ==========================================
// ANIMATION INITIALIZATION
// ==========================================

function initializeAnimations() {
    // Observe elements for scroll-triggered animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all animated elements
    document.querySelectorAll('[class*="fade"], [class*="slide"]').forEach(el => {
        observer.observe(el);
    });
}

// ==========================================
// WAVEFORM ANIMATION
// ==========================================

function initializeWaveformAnimation() {
    const waveform = document.querySelector('.waveform path');
    if (!waveform) return;

    let offset = 0;
    const animate = () => {
        offset += 0.5;
        if (offset > 200) offset = 0;
        
        // Subtle animation effect
        const currentPath = waveform.getAttribute('d');
        
        requestAnimationFrame(animate);
    };

    // Start animation
    animate();
}

// ==========================================
// TREATMENT PLAN INTERACTIONS
// ==========================================

function initializeTreatmentPlan() {
    const treatmentItems = document.querySelectorAll('.treatment-item');
    
    treatmentItems.forEach((item, index) => {
        // Add click interaction
        item.addEventListener('click', () => {
            handleTreatmentClick(item, index);
        });

        // Add hover effect with sound feedback (optional)
        item.addEventListener('mouseenter', () => {
            item.style.transform = 'translateX(4px)';
        });

        item.addEventListener('mouseleave', () => {
            item.style.transform = 'translateX(0)';
        });
    });
}

function handleTreatmentClick(item, index) {
    // Remove active state from all items
    document.querySelectorAll('.treatment-item').forEach(el => {
        el.classList.remove('active');
    });

    // Add active state to clicked item
    item.classList.add('active');

    // Could trigger a modal or detail panel here
    console.log(`Treatment item ${index} clicked`);
    
    // Example: Show treatment details
    showTreatmentDetails(item);
}

function showTreatmentDetails(item) {
    const label = item.querySelector('.treatment-label').textContent;
    
    // You could implement a modal or side panel here
    // For now, we'll just log it
    console.log(`Showing details for: ${label}`);
}

// ==========================================
// VISIT CARDS INTERACTIONS
// ==========================================

function initializeVisitCards() {
    const visitCards = document.querySelectorAll('.visit-card');
    
    visitCards.forEach((card, index) => {
        card.addEventListener('click', () => {
            handleVisitCardClick(card, index);
        });

        // Parallax effect on hover
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;
            
            card.style.transform = `
                translateY(-4px) 
                perspective(1000px) 
                rotateX(${rotateX}deg) 
                rotateY(${rotateY}deg)
            `;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0) perspective(1000px) rotateX(0) rotateY(0)';
        });
    });
}

function handleVisitCardClick(card, index) {
    const title = card.querySelector('.visit-title').textContent;
    const date = card.querySelector('.visit-date').textContent;
    
    console.log(`Visit clicked: ${title} on ${date}`);
    
    // Could open a detailed view or modal
    // showVisitDetails(title, date);
}

// ==========================================
// CONFIDENCE DOTS ANIMATION
// ==========================================

function animateConfidenceDots() {
    const dots = document.querySelectorAll('.confidence-dots .dot');
    
    dots.forEach((dot, index) => {
        setTimeout(() => {
            if (index < 3) { // First 3 dots are active
                dot.classList.add('active');
            }
        }, index * 200);
    });
}

// Start confidence animation on load
setTimeout(animateConfidenceDots, 1000);

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ==========================================
// RESPONSIVE NAVIGATION
// ==========================================

function initializeResponsiveNav() {
    const navMenu = document.querySelector('.nav-menu');
    
    // You could add a hamburger menu for mobile here
    if (window.innerWidth < 768) {
        // Mobile navigation logic
    }
}

window.addEventListener('resize', debounce(() => {
    initializeResponsiveNav();
}, 250));

// ==========================================
// DATA REFRESH (For real implementation)
// ==========================================

function refreshDashboardData() {
    // This would fetch updated data from your backend
    console.log('Refreshing dashboard data...');
    
    // Example:
    // fetch('/api/dashboard-data')
    //     .then(res => res.json())
    //     .then(data => updateDashboard(data))
    //     .catch(err => console.error('Error fetching data:', err));
}

// Auto-refresh every 30 seconds (optional)
// setInterval(refreshDashboardData, 30000);

// ==========================================
// EXPORT FOR TESTING
// ==========================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeAnimations,
        initializeTreatmentPlan,
        initializeVisitCards,
        refreshDashboardData
    };
}
