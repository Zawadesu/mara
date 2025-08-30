let navbar = document.querySelector('.navbar');

document.querySelector('#menu-btn').onclick = () =>{
    navbar.classList.toggle('active');
    searchForm.classList.remove('active');
}

let searchForm = document.querySelector('.search-form');

document.querySelector('#search-btn').onclick = () =>{
    searchForm.classList.toggle('active');
    navbar.classList.remove('active');
}

window.onscroll = () =>{
    navbar.classList.remove('active');
    searchForm.classList.remove('active');
}

const compressedFiles = [
    {
        name: "model",
        type: "zip",
        size: "12 kB",
        date: "2025-6-10",
        path: "model/GPNN.zip",
        description: "MARA predictor model with README file"
    },
    {
        name: "data_log",
        type: "zip",
        size: "129 kB",
        date: "2025-6-10",
        path: "data/data_log.zip",
        description: "Data prepared for MARA in log scale"
    },
    {
        name: "data_zscore",
        type: "zip",
        size: "109 kB",
        date: "2025-6-10",
        path: "data/data_zscore.zip",
        description: "Data standardized by z-score"
    }
];

// rendering
function renderFiles(files) {
    const container = document.getElementById('filesContainer');
    container.innerHTML = '';
    
    files.forEach(file => {
        const card = document.createElement('div');
        card.className = 'file-card';
        
        // icon
        let iconClass = 'zip-icon';
        let iconCode = '<i class="fas fa-file-archive"></i>';
        
        const filePath = `../${file.path}`;

        card.innerHTML = `
            <div class="file-header">
                <div class="file-icon ${iconClass}">${iconCode}</div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-meta">${file.size} · ${file.date}</div>
                </div>
            </div>
            <div class="file-details">
                <p class="file-desc">${file.description}</p>
                <div class="folder-path">Path: ${file.path}</div>
                <a href="${file.path}" class="download-btn" download>
                    <i class="fas fa-download"></i> Download zip
                </a>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Initiation
renderFiles(compressedFiles);

// Order
document.getElementById('sortOptions').addEventListener('change', function(e) {
    const sortBy = e.target.value;
    let sortedFiles = [...compressedFiles];
    
    if (sortBy === 'name') {
        sortedFiles.sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === 'date') {
        sortedFiles.sort((a, b) => new Date(b.date) - new Date(a.date));
    } else if (sortBy === 'size') {
        sortedFiles.sort((a, b) => {
            const sizeA = parseFloat(a.size);
            const sizeB = parseFloat(b.size);
            return sizeB - sizeA;
        });
    }
    
    renderFiles(sortedFiles);
});