import os
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv()

project_endpoint = os.getenv("PROJECT_ENDPOINT")

if not project_endpoint:
    print("PROJECT_ENDPOINT not found in .env")
    exit(1)

print(f"Listing deployments for project: {project_endpoint}")

try:
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )

    deployments = project_client.deployments.list()
    print("\nAvailable Deployments:")
    for deployment in deployments:
        print(f"- Name: {deployment.name}, Model: {deployment.model_name}, Version: {deployment.model_version}")

except Exception as e:
    print(f"Error listing deployments: {e}")
