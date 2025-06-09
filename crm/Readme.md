# Telco Customer VoiceBot

The app interacts with a specified API endpoint to process user inputs and display responses in a chat-like interface. <br>
In order to utilize this app please get your access api key from https://maas.apps.prod.rhoai.rh-aiservices-bu.com/ <br> <br>

**⚠️Recipe for Model As a Service with RHOAI, 3Scale and SSO implementation: [Link](https://github.com/rh-aiservices-bu/models-aas)** <br>

## Prerequisites
- An API key for the Desired Hosted GenAI Model from Model as a Server Backend -> customer-voicebot.py usage
- Access Model Running on RHOAI Inference Server -> customer-voicebot2.py usage

## Create your Service & Route Definitions

```bash
kind: Service
apiVersion: v1
metadata:
  name: service-crm
  namespace: tme-aix
spec:
  ports:
    - protocol: TCP
      port: 15000
      targetPort: 15000
  selector:
    statefulset: tme-aix-wb01 <-- This is your RHOAI WEB Name
```

```bash
kind: Route
apiVersion: route.openshift.io/v1
metadata:
  name: route-crm
  namespace: tme-aix
spec:
  path: /
  to:
    kind: Service
    name: service-crm
    weight: 100
  port:
    targetPort: 15000
  wildcardPolicy: None
```

## Run the app
```bash
app-root) cd Telco-AIX/
(app-root) cd crm/
(app-root) python customer-voicebot.py 
2025-06-05 17:26:50,351 - __main__ - INFO - ==================================================
2025-06-05 17:26:50,351 - __main__ - INFO - Starting Telco CRM VoiceBot - Anti-Simulation Version
2025-06-05 17:26:50,352 - __main__ - INFO - Current time: 2025-06-05 17:26:50
2025-06-05 17:26:50,352 - __main__ - INFO - API URL: http://fnr-tst-tme-aix.apps.sandbox01.narlabs.io/v1/completions
2025-06-05 17:26:50,352 - __main__ - INFO - Static directory: /opt/app-root/src/Telco-AIX/crm/static
2025-06-05 17:26:50,352 - __main__ - INFO - Max context length: 8000 tokens
2025-06-05 17:26:50,352 - __main__ - INFO - Testing API connection...
2025-06-05 17:26:50,352 - __main__ - INFO - Testing connection to: http://fnr-tst-tme-aix.apps.sandbox01.narlabs.io/v1/completions
2025-06-05 17:26:50,736 - __main__ - INFO - Status Code: 200
2025-06-05 17:26:50,737 - __main__ - INFO - Response Headers: {'date': 'Thu, 05 Jun 2025 17:26:49 GMT', 'server': 'uvicorn', 'content-length': '545', 'content-type': 'application/json', 'set-cookie': '9d86516ca3a9a948f57e86e428a097bf=e1d1d7e11ceaefc8998617a9fe4a1669; path=/; HttpOnly'}
2025-06-05 17:26:50,737 - __main__ - INFO - Response Content: {"id":"cmpl-d21ac79f2f9e4f9a8ced051aaeba883d","object":"text_completion","created":1749144410,"model":"mistral-7b-instruct-v03-quantizedw4a16-150","choices":[{"index":0,"text":"\n\nNew to GigSalad\n\nDo you have a band that you'd like to book for an upcoming event? Look no further! The Troublemakers are a dynamic, professional and fun 4-piece band that bring","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":7,"total_tokens":57,"complet...
2025-06-05 17:26:50,737 - __main__ - INFO - JSON Response parsed successfully
2025-06-05 17:26:50,737 - __main__ - INFO - API connection test successful!
2025-06-05 17:26:51,461 - __main__ - INFO - Speech saved to /opt/app-root/src/Telco-AIX/crm/static/welcome.mp3
2025-06-05 17:26:51,461 - __main__ - INFO - Starting Flask development server...
 * Serving Flask app 'customer-voicebot'
 * Debug mode: off
2025-06-05 17:26:51,464 - werkzeug - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:15000
 * Running on http://10.128.1.204:15000
2025-06-05 17:26:51,464 - werkzeug - INFO - Press CTRL+C to quit
```

## Go to Browser and Enjoy The Show! :-)
<div align="center">
    <img src="https://github.com/tme-osx/TME-AIX/blob/main/crm/maas-crm3.png"/>
</div>

## OCP Deployment (No RHOAI-WorkBench Business)
- Build the container image (see Dockerfile here) and push to your image repo <br>
- Edit Deployment.yaml (included here) to have proper image urls and API_Key inside -> just simply;
  
```
oc deploy -f Deployment.yaml
```
