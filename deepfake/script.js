// Redirect to upload page when Get Started is clicked
function goToUploadPage() {
    window.location.href = "upload.html";
}

function processImage() {
    const resultText = document.getElementById("result");
    const fileInput = document.getElementById("imageUpload");

    if (fileInput.files.length === 0) {
        resultText.innerText = "Please upload an image.";
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    resultText.innerText = "Processing...";
    console.log("Sending image to the server...");

    fetch("http://127.0.0.1:8020/upload", {
        method: "POST",
        body: formData,
        headers: {
            "Accept": "application/json"
        }
    })
        .then(response => {
        console.log("Received response:", response);
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        return response.json(); // Parse response JSON
    })
    .then(data => {
        console.log("Parsed JSON data:", data);  // Log the parsed JSON data
        if (data.error) {
            console.error("Error received from server:", data.error);
            resultText.innerText = "Error: " + data.error;
        } else {
            const deepfakeProb = data.deepfake_prob.toFixed(2);
            const realProb = data.real_prob.toFixed(2);
            resultText.innerText = `Deepfake Probability: ${deepfakeProb}%\nReal Probability: ${realProb}%`;
            console.log("Deepfake Probability:", deepfakeProb);
            console.log("Real Probability:", realProb);
        }
    })
        .catch(error => {
        // Check if the error is a result of a network failure or JSON parsing error
        if (error instanceof SyntaxError) {
            resultText.innerText = "Error processing server response. Please check the server logs.";
            console.error("JSON parsing error:", error);
        } else if (error.message.includes("Network response was not ok")) {
            resultText.innerText = "Server responded with an error. Please try again later.";
            console.error("Network error:", error);
        } else {
            resultText.innerText = "Error processing image.";
            console.error("General fetch error:", error);
        }
    });
}
