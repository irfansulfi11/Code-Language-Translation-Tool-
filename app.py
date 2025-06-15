from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import ast
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer with better error handling
model_name = "Salesforce/codet5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    MODEL_LOADED = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL_LOADED = False

# Enhanced code templates with more patterns
CODE_TEMPLATES = {
    'factorial': {
        'python': '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Example usage
print(factorial(5))''',
        
        'java': '''public class Factorial {
    public static void main(String[] args) {
        System.out.println(factorial(5));
    }
    
    public static int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
}''',
        
        'javascript': '''function factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

// Example usage
console.log(factorial(5));''',
        
        'c++': '''#include <iostream>
using namespace std;

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    cout << factorial(5) << endl;
    return 0;
}''',
        
        'c': '''#include <stdio.h>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    printf("%d\\n", factorial(5));
    return 0;
}'''
    },
    
    'fibonacci': {
        'python': '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")''',
        
        'java': '''public class Fibonacci {
    public static void main(String[] args) {
        for (int i = 0; i < 10; i++) {
            System.out.println("F(" + i + ") = " + fibonacci(i));
        }
    }
    
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}''',
        
        'javascript': '''function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Example usage
for (let i = 0; i < 10; i++) {
    console.log(`F(${i}) = ${fibonacci(i)}`);
}''',
        
        'c++': '''#include <iostream>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    for (int i = 0; i < 10; i++) {
        cout << "F(" << i << ") = " << fibonacci(i) << endl;
    }
    return 0;
}''',
        
        'c': '''#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    for (int i = 0; i < 10; i++) {
        printf("F(%d) = %d\\n", i, fibonacci(i));
    }
    return 0;
}'''
    }
}

def detect_code_pattern(code):
    """Detect common coding patterns more accurately"""
    code_lower = code.lower().replace(' ', '').replace('\n', ' ')
    
    # Check for factorial pattern
    if ('factorial' in code_lower or 
        ('n*factorial(n-1)' in code_lower or 'n*factorial(n-2)' in code_lower) or
        ('return1' in code_lower and 'returnn*' in code_lower)):
        return 'factorial'
    
    # Check for fibonacci pattern
    if ('fibonacci' in code_lower or 'fib' in code_lower or
        ('returnn-1' in code_lower and 'returnn-2' in code_lower) or
        ('fibonacci(n-1)+fibonacci(n-2)' in code_lower)):
        return 'fibonacci'
    
    return None

def is_code_corrupted(code):
    """Enhanced corruption detection"""
    if not code or len(code.strip()) < 5:
        return True
    
    # Check for excessive repetition
    repetitive_patterns = [
        r'(.)\1{10,}',  # Same character repeated 10+ times
        r'(\w+)\s+\1\s+\1',  # Same word repeated 3 times
        r'[()]{5,}',  # Too many parentheses
        r'[{}]{5,}',  # Too many braces
        r'[;]{3,}',   # Too many semicolons
    ]
    
    for pattern in repetitive_patterns:
        if re.search(pattern, code):
            logger.warning(f"Corruption detected: pattern {pattern}")
            return True
    
    # Check for malformed syntax
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    if not lines:
        return True
    
    # Check if most lines are malformed
    malformed_count = 0
    for line in lines:
        # Check for lines with unbalanced quotes or brackets
        if (line.count('"') % 2 != 0 or 
            line.count("'") % 2 != 0 or
            abs(line.count('(') - line.count(')')) > 2 or
            abs(line.count('{') - line.count('}')) > 1):
            malformed_count += 1
    
    # If more than 60% of lines are malformed, consider it corrupted
    if len(lines) > 0 and malformed_count / len(lines) > 0.6:
        logger.warning(f"Corruption detected: {malformed_count}/{len(lines)} lines malformed")
        return True
    
    return False

def clean_generated_code(raw_output, original_prompt):
    """Enhanced code cleaning"""
    if not raw_output:
        return ""
    
    code = raw_output.strip()
    
    # Remove the original prompt if it appears in output
    if original_prompt.lower() in code.lower():
        start_idx = code.lower().find(original_prompt.lower())
        if start_idx != -1:
            code = code[start_idx + len(original_prompt):].strip()
    
    # Remove common AI model artifacts
    prefixes_to_remove = [
        "translate", "translation:", "result:", "output:", "code:",
        "here is the", "here's the", "the translated code is:",
        "```", "python", "java", "javascript", "c++", "c"
    ]
    
    for prefix in prefixes_to_remove:
        if code.lower().startswith(prefix.lower()):
            code = code[len(prefix):].strip()
    
    # Remove trailing artifacts
    suffixes_to_remove = ["```", "end", "</code>", "</pre>"]
    for suffix in suffixes_to_remove:
        if code.lower().endswith(suffix.lower()):
            code = code[:-len(suffix)].strip()
    
    # Clean up whitespace
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines at start and end, but keep them in middle for formatting
        if line.strip() or cleaned_lines:
            cleaned_lines.append(line.rstrip())
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def validate_generated_code(code, target_lang):
    """Improved validation with better error reporting"""
    if not code or not code.strip():
        return False, "Empty code generated"
    
    try:
        if target_lang == "python":
            ast.parse(code)
            return True, "Valid Python syntax"
        
        elif target_lang == "java":
            # Basic Java validation
            if 'class ' in code and '{' in code and '}' in code:
                return True, "Basic Java structure present"
            return False, "Missing Java class structure"
        
        elif target_lang == "javascript":
            # Basic JS validation
            if any(keyword in code for keyword in ['function', '=>', 'var ', 'let ', 'const ']):
                return True, "Basic JavaScript structure present"
            return False, "Missing JavaScript function structure"
        
        elif target_lang in ["c++", "c"]:
            # Basic C/C++ validation
            if '#include' in code and 'main' in code:
                return True, f"Basic {target_lang} structure present"
            return False, f"Missing {target_lang} main structure"
        
        return True, "Unknown language, assuming valid"
        
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    translated_code = ""
    translation_method = ""
    
    if request.method == "POST":
        source_lang = request.form.get("source_lang", "").lower().strip()
        target_lang = request.form.get("target_lang", "").lower().strip()
        code_input = request.form.get("code_input", "").strip()

        logger.info(f"Translation request: {source_lang} -> {target_lang}")

        if not all([source_lang, target_lang, code_input]):
            translated_code = "Please fill in all fields."
            translation_method = "Error"
        elif source_lang == target_lang:
            translated_code = "Source and target languages cannot be the same."
            translation_method = "Error"
        else:
            # Try pattern-based translation first
            pattern = detect_code_pattern(code_input)
            logger.info(f"Detected pattern: {pattern}")
            
            if pattern and pattern in CODE_TEMPLATES and target_lang in CODE_TEMPLATES[pattern]:
                translated_code = CODE_TEMPLATES[pattern][target_lang]
                translation_method = f"Template-based translation ({pattern} pattern)"
                logger.info("Used template-based translation")
            
            elif MODEL_LOADED:
                # Try AI-based translation with better parameters
                try:
                    # Create a cleaner, more specific prompt
                    prompt = f"Convert this {source_lang} code to {target_lang}:\n\n{code_input}"
                    
                    # Tokenize with better parameters
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )
                    
                    logger.info(f"Input length: {inputs.input_ids.shape[1]} tokens")
                    
                    # Generate with conservative parameters to reduce corruption
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=600,
                        min_length=20,
                        num_beams=5,  # More beams for better quality
                        early_stopping=True,
                        do_sample=False,  # Disable sampling for more deterministic output
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.5,  # Higher penalty for repetition
                        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                        length_penalty=0.8  # Slight penalty for length
                    )
                    
                    # Decode and clean
                    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"Raw output length: {len(raw_output)}")
                    
                    cleaned_code = clean_generated_code(raw_output, prompt)
                    logger.info(f"Cleaned code length: {len(cleaned_code)}")
                    
                    # Check for corruption
                    if is_code_corrupted(cleaned_code):
                        logger.warning("Generated code is corrupted")
                        raise ValueError("Generated code appears corrupted or malformed")
                    
                    # Validate the code
                    is_valid, validation_msg = validate_generated_code(cleaned_code, target_lang)
                    
                    if is_valid:
                        translated_code = cleaned_code
                        translation_method = f"AI-based translation ({validation_msg})"
                        logger.info("AI translation successful")
                    else:
                        logger.warning(f"Validation failed: {validation_msg}")
                        # Try template fallback if available
                        if pattern and pattern in CODE_TEMPLATES and target_lang in CODE_TEMPLATES[pattern]:
                            translated_code = CODE_TEMPLATES[pattern][target_lang]
                            translation_method = f"Template fallback (AI validation failed: {validation_msg})"
                        else:
                            translated_code = f"AI translation validation failed: {validation_msg}\n\nGenerated code:\n{cleaned_code}\n\nSuggestions:\n1. Check your input code for syntax errors\n2. Try a simpler code example\n3. Use common programming patterns"
                            translation_method = "Validation failed"
                
                except Exception as e:
                    logger.error(f"AI translation failed: {str(e)}")
                    # Template fallback
                    if pattern and pattern in CODE_TEMPLATES and target_lang in CODE_TEMPLATES[pattern]:
                        translated_code = CODE_TEMPLATES[pattern][target_lang]
                        translation_method = f"Template fallback (AI error: {str(e)[:50]}...)"
                    else:
                        translated_code = f"Translation failed: {str(e)}\n\nTry:\n1. Simplifying your code\n2. Checking for syntax errors\n3. Using more common programming patterns"
                        translation_method = "AI translation failed"
            
            else:
                # No model available
                if pattern and pattern in CODE_TEMPLATES and target_lang in CODE_TEMPLATES[pattern]:
                    translated_code = CODE_TEMPLATES[pattern][target_lang]
                    translation_method = "Template-based (AI model unavailable)"
                else:
                    translated_code = "AI model not available. Only template-based translation is supported for factorial and fibonacci patterns."
                    translation_method = "Model unavailable"

    return render_template("index.html", 
                         translated_code=translated_code,
                         translation_method=translation_method)

if __name__ == "__main__":
    app.run(debug=True)