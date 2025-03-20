# E-commerce Customer Support Chatbot

A conversational AI agent designed to handle e-commerce customer support inquiries, including order status checks, return policy information, and human handoff for complex issues.

## Features

- **Order Status Lookup**: Check the status of orders using order ID.
- **Return Policy Information**: Provide details on return policies for different product categories.
- **Human Handoff**: Collect customer information and escalate to human representatives.
- **Offline Capability**: Function with cached data when connectivity is limited.

## Project Structure
``` ecommerce-chatbot/

├── src/
│   ├── chatbot.py              # Main application with Langgraph implementation
│   ├── utils.py               # Helper functions for data processing
│   └── config.py              # Configuration settings
├── data/
│   ├── olist_orders_dataset.csv  # Original dataset
│   ├── cached_orders.csv      # Cached data for offline use
│   └── cached_policies.json   # Cached return policies
├── tests/
│   └── test_chatbot.py        # Unit tests with predefined dialogues
├── docs/
│   └── evaluation_report.md   # Detailed evaluation with metrics
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions workflow for CI/CD
├── .env                       # API keys (gitignored)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```
## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Gemini AI API

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/moses-y/ecommerce-chatbot.git
   cd ecommerce-chatbot
    ```
2. **Install dependencies**:
    ```bash
    # Install uv first
    pip install uv

    # Then use uv to install dependencies
    uv pip install -r requirements.txt
    ```
# Alternative method to resolve dependency issues (for Windows PowerShell users):

### PowerShell security restriction: Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Create a fresh virtual environment:
```bash
# Create a new virtual environment
python -m venv fresh_env

# Activate it (Windows)
fresh_env\Scripts\activate

# Install uv in the new environment
pip install uv

# Install numpy 1.26.4 first
uv pip install numpy==1.26.4

# Verify numpy version
python -c "import numpy; print(numpy.__version__)"

# Install other dependencies
uv pip install python-dotenv==1.0.1 langchain==0.3.0 langchain-core==0.3.46 langgraph==0.3.14 langchain-openai==0.3.9 langchain-google-genai==2.1.0 google-ai-generativelanguage==0.6.17 pytest==7.4.0 gradio==5.22.0

# Install pandas last
- uv pip install pandas==2.2.3
```

## Create a .env file in the root directory with your API keys:
- OPENAI_API_KEY=your_openai_api_key_here
- GOOGLE_API_KEY=your_google_api_key_here

## Place the Olist orders dataset in the data directory:
- data/olist_orders_dataset.csv

Download it from Kaggle `https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv.`

## Running the Chatbot
To start the chatbot in interactive mode:

``` bash
python src/chatbot.py
```
## Running the Web Interface

To start the chatbot with the Gradio web interface:
```bash
python app.py
```
This will launch a local web server and provide a URL (e.g., http://127.0.0.1:7860) you can open in your browser. The interface allows you to:
- Chat with the e-commerce support bot.
- Ask about order status using order IDs from the dataset.
- Inquire about return policies.
- Request to speak with a human representative.
## Testing
Run the test suite to verify functionality:
```bash
pytest tests/
```
## Usage Examples
Checking Order Status

User: What's the status of my order?

Assistant: I'd be happy to help check your order status. Could you please provide your order ID?

User: ORDER123

Assistant: Order #ORDER123 is currently delivered.

```Your order has been delivered to the specified address.

Purchase date: January 15, 2023
Approved date: January 15, 2023
Estimated delivery: January 25, 2023
Delivered on: January 22, 2023
```
### Return Policy Information

User: What's your return policy?

Assistant: You can return most items within 30 days of purchase for a full refund or exchange. Items must be in their original condition, with all tags and packaging intact. 
Please bring your receipt or proof of purchase when returning items.

### Different product categories may have specific policies:

Electronics: 14-day return window
Clothing: 45-day return window
Perishable items: Must be reported within 3 days if damaged

Would you like more information about a specific category?

## Human Representative Request

User: I need to speak to a human

Assistant: I understand you'd like to speak with a human representative. I'll help connect you.

Could you please provide your name?

User: John Smith

Assistant: Thank you, John Smith. Could you please provide your email address?

User: john.smith@example.com

Assistant: Thank you. Finally, could you please provide your phone number?

User: 555-123-4567

Assistant: Thank you for providing your information. A customer service representative will contact you soon at john.smith@example.com or 555-123-4567. Is there anything else you'd like to add before I submit your request?
## Evaluation
The chatbot has been evaluated based on the following criteria:
- Code Quality: Clarity, organization, and adherence to best practices.
- Functionality: Accuracy and completeness of implemented features.
- Performance: Efficiency and responsiveness.
- User Interface: Natural conversation flow and helpfulness.
- Offline Capability: Effective data caching and offline access.

`For detailed evaluation metrics and test dialogues, see the Evaluation Report (docs/evaluation_report.md).`
## Future Improvements
- Enhanced order tracking with shipping API integration.
- Multilingual support.
- Personalized responses based on customer history.
- Proactive notifications for delivery issues.
- Analytics integration for identifying common customer issues.
## License
MIT

---
