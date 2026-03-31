document.addEventListener('click', function(e) {
    const anchor = e.target.closest('a');
    
    if (anchor && anchor.href && anchor.href.startsWith('http')) {
        // Sirf external links ko check karo, internal scrolling ko nahi
        if (anchor.href.includes(window.location.hostname)) return;

        e.preventDefault(); 
        const targetUrl = anchor.href;

        console.log("SafeStride AI scanning link...");

        chrome.runtime.sendMessage({ action: "checkLink", url: targetUrl }, (response) => {
            if (response && response.isScam) {
                const confirmVisit = confirm(`⚠️ WARNING BY SAFESTRIDE AI!\n\nYe link SCAM lag raha hai.\nReason: ${response.reason}\n\nKya aap phir bhi is risk bhari site par jana chahte hain?`);
                if (confirmVisit) {
                    window.location.href = targetUrl;
                }
            } else {
                window.location.href = targetUrl;
            }
        });
    }
}, true);
