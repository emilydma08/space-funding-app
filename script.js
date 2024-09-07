// JavaScript to add interactivity
document.addEventListener('DOMContentLoaded', function() {
    console.log('Website loaded successfully!');
    
    // Example: Scroll to section on navigation click
    const links = document.querySelectorAll('nav ul li a');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.getAttribute('href').substring(1);
            const section = document.getElementById(sectionId);
            window.scrollTo({
                top: section.offsetTop - 50,
                behavior: 'smooth'
            });
        });
    });
});
