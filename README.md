# Final Year Project: System Log Anomaly Detection (COMP6024)

This is my final year individual project. I built a Python tool that uses machine learning to scan through system logs and automatically flag weird or suspicious activity. 

## What the Project Does
Manually reading through thousands of server logs to find security issues or system crashes is impossible. This project solves that by using an AI model to look at the logs, learn what "normal" activity looks like, and call out the unusual things that need a human to check them out.

## How the Code Works
1. **Cleaning the Data:** The script reads raw CSV log files, fixes any missing information, and organizes the timestamps, usernames, and messages.
2. **Finding Patterns:** It pulls out details like what time of day actions happened, how often a specific user logs in, and how long the log messages are.
3. **The AI Model:** It uses an **Isolation Forest** algorithm. Because the logs aren't pre-labeled as "good" or "bad", this unsupervised model finds anomalies by isolating the rare data points that look completely different from everything else.
4. **The Output:** It sorts everything and saves the most suspicious alerts into a file called `ranked_anomalies.csv`.

## How to Run It
1. Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install pandas scikit-learn
   ```
2. Put your log file in the folder and name it `system_logs.csv`.
3. Run the main script:
   ```bash
   python anomaly_detection.py
   ```
