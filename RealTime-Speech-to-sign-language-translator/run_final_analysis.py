import subprocess
import sys
import os

os.chdir(r'c:\Users\Dell\Desktop\speech-sign-avatar')
result = subprocess.run([sys.executable, 'analyze_evaluation_final.py'])
sys.exit(result.returncode)
