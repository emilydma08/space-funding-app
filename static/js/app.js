const questions = [
    {
        question: "How do you spell 'space'?",
        answers: [
            { text: "Berlin", correct: false },
            { text: "space", correct: true },
            { text: "Berlin", correct: false },
            { text: "Berlin", correct: false }
        ]
    },
    {
        question: "Which one is a planet?",
        answers: [
            { text: "Sun", correct: false },
            { text: "Earth", correct: true },
            { text: "Moon", correct: false },
            { text: "Comet", correct: false }
        ]
    }
];

let currentQuestionIndex = 0;

document.addEventListener("DOMContentLoaded", () => {
    loadQuestion(currentQuestionIndex);
});

function loadQuestion(index) {
    const questionElement = document.getElementById('question');
    const answerButtons = document.getElementById('answer-buttons');

    // Clear previous answers
    answerButtons.innerHTML = '';

    // Load the question
    const currentQuestion = questions[index];
    questionElement.innerText = currentQuestion.question;

    // Load the answers
    currentQuestion.answers.forEach(answer => {
        const button = document.createElement('button');
        button.innerText = answer.text;
        button.addEventListener('click', () => selectAnswer(button, answer.correct));
        answerButtons.appendChild(button);
    });
}

function selectAnswer(button, correct) {
    // Mark the selected answer
    button.classList.add(correct ? 'correct' : 'incorrect');

    // Move to the next question after 1 second
    setTimeout(() => {
        currentQuestionIndex++;

        // Check if there are more questions
        if (currentQuestionIndex < questions.length) {
            loadQuestion(currentQuestionIndex);
        } else {
            alert("Trivia finished!");
            currentQuestionIndex = 0; // Reset the quiz if needed
            loadQuestion(currentQuestionIndex);
        }
    }, 1000);
}
