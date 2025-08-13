// Keywords tracking visualization for Hellenic Parliament Speech Analysis

document.addEventListener('DOMContentLoaded', function() {
    const keywordInput = document.getElementById('keyword-input');
    const addKeywordButton = document.getElementById('add-keyword');
    const keywordTagsContainer = document.getElementById('keyword-tags');
    const trackingView = document.getElementById('tracking-view');
    const keywordChart = document.getElementById('keyword-chart');

    if (!keywordInput || !addKeywordButton || !keywordTagsContainer || !trackingView || !keywordChart) return;

    // Keywords tracking state
    let trackedKeywords = [];
    let chart = null;

    // Initialize Chart.js instance
    function initializeChart() {
        const ctx = keywordChart.getContext('2d');

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['1989', '1995', '2000', '2005', '2010', '2015', '2020'],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Year'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency (%)'
                        },
                        min: 0
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Keyword Frequency Over Time'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 12
                        }
                    }
                }
            }
        });
    }

    // Initialize the chart on page load
    initializeChart();

    // Add keyword button click handler
//    addKeywordButton.addEventListener('click', function() {
//        addKeyword();
//    });

    // Add keyword on Enter key
    keywordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            addKeyword();
        }
    });

    // Tracking view change handler
    trackingView.addEventListener('change', function() {
        updateChart();
    });

    // Function to add a keyword to tracking
    function addKeyword() {
        const keyword = keywordInput.value.trim();

        if (keyword === '') return;

        // Check if keyword already exists
        if (trackedKeywords.includes(keyword)) {
            alert('This keyword is already being tracked');
            return;
        }

        // Add to tracked keywords
        trackedKeywords.push(keyword);

        // Create tag element
        createKeywordTag(keyword);

        // Clear input
        keywordInput.value = '';

        // Update chart
        updateChart();

        // Focus on input for next entry
        keywordInput.focus();
    }

    // Create a visual tag for the keyword
    function createKeywordTag(keyword) {
        const tag = document.createElement('div');
        tag.className = 'keyword-tag';

        // Generate a random color for this keyword
        const color = getRandomColor();
        tag.style.backgroundColor = color;

        tag.innerHTML = `
            <span>${keyword}</span>
            <button class="remove-tag" data-keyword="${keyword}">✕</button>
        `;

        keywordTagsContainer.appendChild(tag);

        // Add remove event listener
        tag.querySelector('.remove-tag').addEventListener('click', function() {
            const keywordToRemove = this.dataset.keyword;
            removeKeyword(keywordToRemove);
            tag.remove();
        });
    }

    // Update the chart with current keywords and view mode
    function updateChart() {
        if (!chart) return;

        const viewMode = trackingView.value;

        // Clear existing datasets
        chart.data.datasets = [];

        // Add a dataset for each tracked keyword
        trackedKeywords.forEach((keyword, index) => {
            // Generate mock data based on keyword and view mode
            const data = generateMockData(keyword, viewMode);

            chart.data.datasets.push({
                label: keyword,
                data: data,
                borderColor: getRandomColor(),
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.3,
                pointRadius: 4
            });
        });

        chart.update();
    }

    // Generate a random color for chart lines
    function getRandomColor() {
        const colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#d35400', '#34495e', '#16a085', '#c0392b'
        ];

        return colors[Math.floor(Math.random() * colors.length)];
    }

    // KEYWORDS BY YEAR BUTTON
    addKeywordButton.addEventListener("click", () => {
        const name = keywordInput.value.trim().toLowerCase();
        const type = trackingView.value;

        if (!name) {
            alert("Δώσε όνομα βουλευτή ή κόμματος.");
            return;
        }

        fetch("/keywords/by_year", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ type: type, name: name })
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            keywordInput.innerHTML = "";
            if (data.error) {
                keywordChart.innerHTML = "<p>Δε βρέθηκαν δεδομένα.</p>";
                return;
            }

            for (const year in data) {
                const p = document.createElement("p");
                p.innerHTML = `<strong>${year}:</strong><br>`;
                data[year].forEach(obj => {
                    p.innerHTML += `&nbsp;&nbsp;• ${obj.keyword} (score: ${obj.score})<br>`;
                });
                keywordChart.appendChild(p);
            }
        });
    });
});
