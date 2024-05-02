

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

};


let userInput = document.getElementById('transcript');
userInput.addEventListener('submit', handleResponse);