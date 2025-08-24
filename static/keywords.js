// Keywords tracking visualization for Hellenic Parliament Speech Analysis

document.addEventListener('DOMContentLoaded', function() {
    const keywordInput = document.getElementById('keyword-input');
    const addKeywordButton = document.getElementById('add-keyword');
    const keywordTagsContainer = document.getElementById('keyword-tags');
    const trackingView = document.getElementById('tracking-view');
    const keywordChart = document.getElementById('keyword-chart');

    if (!keywordInput || !addKeywordButton || !keywordTagsContainer || !trackingView || !keywordChart) return;



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
