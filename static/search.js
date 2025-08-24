// Κάνει ένα απλό search χωρίς φίλτρα προς στιγμήν
    // Search functionality for Hellenic Parliament Speech Analysis Platform

document.addEventListener('DOMContentLoaded', function() {
   const searchForm = document.getElementById('search-form');
   const searchInput = document.getElementById('search-input');
   const searchResults = document.getElementById('search-results');
   const dateRangeFilter = document.getElementById('date-range');
   const partyFilter = document.getElementById('party-filter');
   const mpFilter = document.getElementById('mp-filter');

   if (!searchForm || !searchResults) return;
    // Search form submission handler
   searchForm.addEventListener('submit', function(e) {
       e.preventDefault();
       const searchTerm = searchInput.value.trim().toLowerCase();
       if (searchTerm === '') {
           searchResults.innerHTML = '<p class="placeholder-text">Enter search terms above to see results</p>';
           return;
       }

       // Get filter values
       const dateRange = dateRangeFilter.value;
       const party = partyFilter.value;
       const mp = mpFilter.value;

       //------------από Βάλια-----------
       fetch("/search", {
           method: "POST",
           headers: {
               "Content-Type": "application/json"
           },
           body: JSON.stringify({ query: searchTerm })
       })
       .then(response => response.json())
       .then(data => {
           searchResults.innerHTML = "";
           if (data.length === 0) {
               searchResults.innerHTML = "<p>No results found.</p>";
           } else {
               console.log(data);
               data.forEach(item => {
                   const div = document.createElement("div");
                   div.classList.add("result");
                   div.innerHTML = `
                       <h3>${item.member} (${item.party}) - ${item.date}</h3>
                       <p><strong>Score:</strong> ${item.score}</p>
                       <p>${item.speech}</p>
                   `;

                   div.style.backgroundColor = "white";
                   div.style.padding = "var(--spacing-lg)";
                   div.style.margin = "5px";
                   div.style.borderRadius = "var(--radius-md)";
                   div.style.boxShadow = "0 2px 10px rgba(0, 0, 0, 0.1)";
                   div.style.minHeight = "200px";

                   searchResults.appendChild(div);
               });
           }
       });
       //-------------------------------------

// <!-- >       // Perform search with filters-->
// <!--  >     const results = performSearch(searchTerm, dateRange, party, mp);-->
// <!--        // Display results-->
// <!--        displaySearchResults(results);-->
   });
});
