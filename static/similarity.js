document.addEventListener('DOMContentLoaded', function() {
    const memberInput = document.getElementById('name-input');
    const addMemberButton = document.getElementById('add-name');
    const similarityChart = document.getElementById('similarity-chart');

    addMemberButton.addEventListener("click", () => {
        const member = memberInput.value.trim().toLowerCase();
        console.log(member);
        if (!member) {
            alert("Δώσε όνομα βουλευτή.");
            return;
        }

        fetch("/similarity", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ member: member })
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            memberInput.innerHTML = "";
            if (data.error) {
                similarityChart.innerHTML = "<p>Δε βρέθηκαν δεδομένα.</p>";
                return;
            }

//            for (const mem in data) {
//                const p = document.createElement("p");
//                p.innerHTML = `<strong>${mem}:</strong><br>`;
//                data[mem].forEach(obj => {
//                    p.innerHTML += `&nbsp;&nbsp;• ${obj.member} (score: ${obj.score})<br>`;
//                });
//                similarityChart.appendChild(p);
//            }
        });
    });
});
