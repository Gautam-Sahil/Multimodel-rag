
import json
from datetime import datetime

class RAGEvaluator:
    def __init__(self, qa_system):
        self.qa_system = qa_system
        self.test_cases = [
            {
                "query": "What are the key findings in the executive summary?",
                "type": "text",
                "expected_keywords": ["summary", "findings", "conclusion"]
            },
            {
                "query": "Show me data from any tables in the document",
                "type": "table",
                "expected_keywords": ["table", "data", "figure"]
            },
            {
                "query": "What information is contained in the images?",
                "type": "image",
                "expected_keywords": ["image", "chart", "figure"]
            }
        ]
    
    def run_evaluation(self):
        results = []
        for test in self.test_cases:
            response = self.qa_system.answer(test["query"])
            
            # Simple keyword check
            keyword_score = 0
            for keyword in test["expected_keywords"]:
                if keyword.lower() in response.lower():
                    keyword_score += 1
            
            results.append({
                "query": test["query"],
                "type": test["type"],
                "response": response[:200],  # First 200 chars
                "keyword_score": f"{keyword_score}/{len(test['expected_keywords'])}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Save results
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results