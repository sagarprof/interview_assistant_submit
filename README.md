# Gen AI Interview Assistant

A sophisticated gen ai recruitment and analysis application that uses Google AI agents to evaluate candidates against job descriptions. The system performs comprehensive analysis, skill mapping, question generation, and suitability predictions.

## Features

### 1. **Resume & Job Description Analysis**
- Upload candidate resumes (PDF or TXT)
- Upload job descriptions (PDF or TXT)
- Automatic text extraction and processing

### 2. **Multi-Agent Pipeline**
The application uses a sequential agent swarm with four specialized agents:

#### Agent 1: Analyzer
- Compares resume against job description
- Identifies matching requirements and skill gaps
- Provides overall fit percentage (0-10)
- Structured output with strengths and gaps

#### Agent 2: Skill Mapper
- Detects 15+ AI/Technical skills including:
  - Prompt Engineering
  - RAG (Retrieval Augmented Generation)
  - LLM Fine-tuning & Inference
  - Vector Databases
  - Machine Learning (NLP, CV)
  - API Development
  - Cloud Deployment
  - Frontend Development
- Verifies skills with evidence from resume
- Handles skill synonyms and related technologies

#### Agent 3: Question Generator
- Generates 10 advanced technical interview questions
- Questions tailored to candidate's background
- Includes detailed answer hints with keywords
- Uses Google Search for latest information

#### Agent 4: Synthesis Agent
- Provides executive summary of candidate fit
- Predicts candidate's experience tilt (Data Scientist, DevOps, Test Engineer, Data Engineer)
- Assesses level of understanding (Beginner, Intermediate, Expert)
- Evaluates learning interest based on certifications, courses, blogs, papers
- Generates readiness score (0-10)

### 3. **Interactive Dashboard**
- **Metrics Display**: Skill fit, JD fit scores
- **Questions Tab**: View generated interview questions with answer hints
- **Skill Mapping Tab**: Detailed AI skills verification
- **Requirement Match Tab**: Strengths and gaps analysis
- **Suitability Prediction Tab**: Candidate fit summary
- **Experience Analysis Tab**: Tilted experience assessment
- **Learning Interest Tab**: Passion and growth potential analysis

### 4. **PDF Report Generation**
- Downloadable comprehensive candidate guide
- Includes all analysis results, skills, and interview questions
- Professional formatting with tables and sections

## Installation

### Prerequisites
- Python 3.8+
- Google API Key for Gemini and Search APIs

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd interview_assistant_submit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. **Run the application**
   ```bash
   streamlit run interwiver_assistant.py
   ```

## Usage

1. **Open the application** in your browser (typically `http://localhost:8501`)

2. **Upload Files**
   - Upload candidate resume (PDF or TXT)
   - Upload job description (PDF or TXT)

3. **Analyze**
   - Click "Analyze & Generate Guide" button
   - Wait for all agents to complete processing

4. **Review Results**
   - View metrics (Skill Fit, JD Fit)
   - Navigate through tabs for detailed analysis
   - Review generated interview questions

5. **Download Report**
   - Click "Download Report" to get PDF document
   - Share with recruiting team

## Project Structure

```
interview_assistant_submit/
├── interwiver_assistant.py      # Main application file
├── utils.py                      # UI styling utilities
├── requirements.txt              # Project dependencies
├── __init__.py                  # Package initialization
└── README.md                    # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| PyPDF2 | PDF text extraction |
| reportlab | PDF report generation |
| pydantic | Data validation and schemas |
| python-dotenv | Environment variable management |
| google-adk | Google AI agents framework |
| google-genai | Google GenAI API client |
| pandas | Data manipulation and display |

## Data Models

### AnalysisOutput
```python
- candidate_summary: str
- match_score: Match
  - overall_percentage: int (0-10)
  - match: List[str]
  - gaps: List[str]
```

### SkillMappingOutput
```python
- skills: List[SkillItem]
  - skill: str
  - present: bool
  - related_skills: str
```

### QuestionOutput
```python
- technical_questions: List[QuestionItem]
  - question: str
  - answer_hints: List[str]
```

### SynthesisOutput
```python
- summary: str
- Tilted_experience: str
- level_of_understanding: str
- learning_interest: str
- readiness_score: int (0-10)
```

## Key Features Explained

### Smart Text Processing
- Robust JSON extraction from LLM outputs
- Handles malformed responses and markdown formatting
- Automatic data structure normalization

### Skill Detection with Synonyms
- Recognizes alternate names (e.g., "Language Models" = "GenAI", "Postgres" = "SQL")
- Quotes relevant evidence from resume
- Handles missing or unverified skills gracefully

### Interview Question Generation
- Context-aware questions based on resume
- Mix of theoretical and practical questions
- Includes detailed answer guidelines for evaluators

### Comprehensive Reporting
- Executive summary with key metrics
- Match analysis with strengths/gaps
- Skill verification table
- Interview questions with answer hints

## UI/UX Features

- **Dark Theme**: Modern dark interface with purple accents
- **Responsive Layout**: Multi-column design for optimal viewing
- **Interactive Tabs**: Easy navigation between analysis sections
- **Progress Indicators**: Real-time status updates during processing
- **Styled Cards**: Professional metric display with borders and shadows
- **Color-Coded Analysis**: Green for matches, red for gaps

## Error Handling

The application includes robust error handling for:
- Invalid PDF/text files
- Missing environment variables
- Malformed JSON responses from agents
- Data structure inconsistencies

## Performance Considerations

- Text truncation for large documents (3000 chars for resume, 2000 for JD)
- Async processing for agent pipeline
- In-memory session management
- Efficient PDF generation

## Future Enhancements

- Support for multiple resume formats
- Candidate comparison functionality
- Custom skill library management
- Template-based question generation
- Integration with ATS systems
- Email report delivery

## Troubleshooting

### Missing API Key
- Ensure `GOOGLE_API_KEY` is set in `.env` file
- Verify API key has access to Gemini and Search APIs

### PDF Upload Issues
- Check file format (PDF or TXT)
- Ensure file is not corrupted
- Try extracting text manually if needed

### Agent Timeout
- Check internet connection
- Verify API quotas are not exceeded
- Try with smaller documents

## License

This project is for recruitment and candidate assessment purposes.

## Support

For issues or questions, contact the development team.
