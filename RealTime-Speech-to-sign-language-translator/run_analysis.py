import subprocess
import sys

result = subprocess.run([sys.executable, r'c:\Users\Dell\Desktop\speech-sign-avatar\analyze_evaluation.py'], 
                       cwd=r'c:\Users\Dell\Desktop\speech-sign-avatar')
sys.exit(result.returncode)
