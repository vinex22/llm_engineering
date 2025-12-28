from azure.identity import AzureCliCredential
import sys

try:
    cred = AzureCliCredential()
    token = cred.get_token("https://management.azure.com/.default")
    print("Successfully acquired token via Azure CLI")
except Exception as e:
    print(f"Failed to acquire token: {e}")
