region_extract_prompt = """I am providing you with an abstract description from a Wikipedia page about a music artist. Your task is to identify the region, country, or cultural background in which this artist is most active or is closely associated with.

Here is the abstract:

{row['abstract']}

Please provide a concise response with the identified region, country, or culture."""

question_generation_prompt = """You are an AI assistant tasked with generating **two multiple-choice questions** based on a given section of text from a Wikipedia page about a music artist, specifically focusing on their {row['topic']}. Your goal is to create **one basic factual recall question** and **one challenging interpretative or analytical question**, following these guidelines:

### **Question Requirements**
1. **Factual Recall Question**  
   - Should be simple or moderately difficult.  
   - Must ask for a straightforward fact directly stated in the text.  

2. **Interpretative/Analytical Question**  
   - Should require deeper thinking, analysis, or inference based on the text.  
   - Must encourage critical engagement with the information provided.  

### **Answer Choice Guidelines**
- Each question must have **one correct answer** and **three plausible distractors**.  
- Ensure that the distractors are realistic but clearly distinguishable from the correct answer.  
- **Do not use pronouns; instead, explicitly use the artist's name ({row['Artist']}) in the questions and answer choices.**  

### **Output Format**
Ensure the response follows this exact format:
Question 1: [Factual recall question]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
Answer 1: [A/B/C/D]

Question 2: [Interpretative/Analytical question]
A. [Option 1]
B. [Option 2]
C. [Option 3]
D. [Option 4]
Answer 2: [A/B/C/D]

The questions must be **clear, concise, factually accurate**, and directly derived from the text. Avoid ambiguity and ensure that the correct answers are unambiguous."""