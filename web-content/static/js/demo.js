function formatText(text) {
    // Replace newlines with <br> tags
    return text.trim().replace(/\n/g, '<br>');
}

function handleResponse(event) {
    event.preventDefault();
    let inputText = event.target.elements.query.value;
    console.log(inputText);
    fetch('./generate_output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({response: inputText})
    }).then(response => response.json()).then(data => generate_content(data))
};

function generate_content(data) {
    console.log(data)
    let sentimentComponent = document.querySelector('.sentiment-result')
    sentimentComponent.textContent = data['sentiment'].charAt(0).toUpperCase() + data['sentiment'].slice(1);
    console.log(sentimentComponent.textContent)

    let summaryComponent = document.querySelector('.summary-result')
    console.log(summaryComponent)
    summaryComponent.innerHTML = formatText(data["summary"])
};


let userInput = document.getElementById('transcript');
userInput.addEventListener('submit', handleResponse);