const themeSwitch = document.getElementById('theme-switch');
const body = document.body;
const container = document.querySelector('.container');
const navItems = document.querySelectorAll('.nav-item');
const buttons = document.querySelectorAll('button');
const footer = document.querySelector('.footer');
const heading = document.querySelector('h1');
const toggleLabel = document.querySelector('.toggle-label');
const settingsBtn = document.querySelector('.settings-btn'); // Added this line

const applyDarkMode = (isDark) => {
    if (isDark) {
        body.classList.add('dark-mode');
        container.classList.add('dark-mode');
        footer.classList.add('dark-mode');
        navItems.forEach(item => item.classList.add('dark-mode'));
        buttons.forEach(button => button.classList.add('dark-mode'));
        heading.classList.add('dark-mode');
        toggleLabel.classList.add('dark-mode');
        settingsBtn.classList.add('dark-mode'); // Added this line
        localStorage.setItem('theme', 'dark');
    } else {
        body.classList.remove('dark-mode');
        container.classList.remove('dark-mode');
        footer.classList.remove('dark-mode');
        navItems.forEach(item => item.classList.remove('dark-mode'));
        buttons.forEach(button => button.classList.remove('dark-mode'));
        heading.classList.remove('dark-mode');
        toggleLabel.classList.remove('dark-mode');
        settingsBtn.classList.remove('dark-mode'); // Added this line
        localStorage.setItem('theme', 'light');
    }
};

// Check for saved theme preference in localStorage
const currentTheme = localStorage.getItem('theme');
if (currentTheme === 'dark') {
    applyDarkMode(true);
    themeSwitch.checked = true;
}

// Toggle dark mode on checkbox click
themeSwitch.addEventListener('change', () => {
    applyDarkMode(themeSwitch.checked);
});
