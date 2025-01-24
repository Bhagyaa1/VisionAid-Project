import requests
import re

# URLs
passage_url = "https://quest.squadcast.tech/api/RA2111003011617/worded_ip"
submission_url_template = "https://quest.squadcast.tech/api/RA2111003011617/submit/worded_ip?answer={}&extension=py"

# Dictionary to map word numbers to digits
word_to_num = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

def fetch_passage():
    response = requests.get(passage_url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Error fetching the passage")

def parse_ip_address(passage):
    # Regular expression to find the IP address pattern in words
    ip_pattern = r"(zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+point\s+(zero|one|two|three|four|five|six|seven|eight|nine)){3}"
    matches = re.findall(ip_pattern, passage)
    
    # Convert matched words to numbers and format as an IP address
    if matches:
        ip_address = []
        for word_tuple in matches:
            num = "".join(word_to_num[word] for word in word_tuple)
            ip_address.append(num)
        return ".".join(ip_address)
    else:
        raise ValueError("No IP address found in passage")

def submit_answer(answer, code):
    submission_url = submission_url_template.format(answer)
    response = requests.post(submission_url, data=code)
    if response.status_code == 200:
        print("Submission successful:", response.json())
    else:
        print("Submission failed:", response.json())

# Main function to execute the program
def main():
    # Fetch passage
    passage = fetch_passage()
    
    # Parse the IP address from the passage
    answer = parse_ip_address(passage)
    
    # Prepare the code for submission (replace <place your entire code here> with the actual code if needed)
    code = """import requests
import re

# URLs
passage_url = 'https://quest.squadcast.tech/api/RA2111003011617/worded_ip'
submission_url_template = 'https://quest.squadcast.tech/api/RA2111003011617/submit/worded_ip?answer={}&extension=py'

# Dictionary to map word numbers to digits
word_to_num = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}

def fetch_passage():
    response = requests.get(passage_url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception('Error fetching the passage')

def parse_ip_address(passage):
    # Regular expression to find the IP address pattern in words
    ip_pattern = r'(zero|one|two|three|four|five|six|seven|eight|nine)(?:\\s+point\\s+(zero|one|two|three|four|five|six|seven|eight|nine)){3}'
    matches = re.findall(ip_pattern, passage)
    
    # Convert matched words to numbers and format as an IP address
    if matches:
        ip_address = []
        for word_tuple in matches:
            num = ''.join(word_to_num[word] for word in word_tuple)
            ip_address.append(num)
        return '.'.join(ip_address)
    else:
        raise ValueError('No IP address found in passage')

# Main function to execute the program
def main():
    # Fetch passage
    passage = fetch_passage()
    
    # Parse the IP address from the passage
    answer = parse_ip_address(passage)
    
    # Submit the answer
    submit_answer(answer, <code>)
    
if __name__ == '__main__':
    main()"""
    
    # Submit the answer
    submit_answer(answer, code)

if __name__ == "__main__":
    main()
