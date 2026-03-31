chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "checkLink") {
        fetch("http://localhost:3000/check-link", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: request.url })
        })
        .then(res => res.json())
        .then(data => {
            sendResponse({ isScam: data.isScam, reason: data.reason });
        })
        .catch(err => {
            console.error("Backend Server Error:", err);
            sendResponse({ isScam: false }); 
        });
        return true; 
    }
});
