// General main.js for HastaVeda Portal
document.addEventListener("DOMContentLoaded", () => {
    console.log("[HastaVeda] Classical Bharatanatyam Practice Companion initialized successfully!");
    
    // Add page transition effects softly
    const mainContent = document.querySelector("main");
    if (mainContent) {
        mainContent.style.opacity = 0;
        mainContent.style.transition = "opacity 0.5s ease";
        setTimeout(() => {
            mainContent.style.opacity = 1;
        }, 50);
    }

    // Highlight current active page link if missing in templates
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll(".navbar-bharata .nav-link");
    navLinks.forEach(link => {
        const href = link.getAttribute("href");
        if (href && currentPath === href) {
            link.parentElement.classList.add("active");
        }
    });
});
