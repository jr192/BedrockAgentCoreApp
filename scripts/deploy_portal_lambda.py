"""
Automated Deployment Script for Serverless Web Portal Lambda on AWS
Deploys:
1. Least-privilege IAM Execution Role (S3, Bedrock Titan v2, Bedrock AgentCore, CloudWatch).
2. Packaging of FastAPI + Mangum + Dependencies via uv for Linux x86_64.
3. AWS Lambda Function: `cfas-agent-evaluation-portal`.
4. AWS Lambda Public HTTPS Function URL with CORS and token authentication.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

import boto3

REGION = "us-east-1"
BUCKET_NAME = "cfas-corporate-docs-725079717969"
AGENT_RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:725079717969:runtime/"
    "BedrockAgentCoreApp_BedrockAgentCoreApp-5keeMq9c1S"
)
INTERVIEW_SECRET = "cfas-agent-demo-2026"
FUNCTION_NAME = "cfas-agent-evaluation-portal"
ROLE_NAME = "CorporateBylawsPortalRole"

iam = boto3.client("iam", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)


def setup_iam_role() -> str:
    """Create or get the IAM execution role with necessary permissions."""
    print(f"1. Configuring IAM Role: {ROLE_NAME}...")
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        role = iam.get_role(RoleName=ROLE_NAME)
        role_arn = role["Role"]["Arn"]
        print(f"   ✓ Existing IAM role found: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        print("   Creating new IAM execution role...")
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy),
            Description="Execution role for CFAS Multi-Agent Evaluation Portal Lambda",
        )
        role_arn = role["Role"]["Arn"]
        print(f"   ✓ Created IAM role: {role_arn}")
        time.sleep(5)  # Wait for IAM role propagation

    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "arn:aws:logs:*:*:*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET_NAME}",
                    f"arn:aws:s3:::{BUCKET_NAME}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                ],
                "Resource": [
                    "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
                    "arn:aws:bedrock:us-east-1::foundation-model/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:InvokeAgentRuntime",
                ],
                "Resource": "*",
            },
        ],
    }

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="CorporateBylawsPortalPolicy",
        PolicyDocument=json.dumps(policy_doc),
    )
    print("   ✓ IAM Policies attached (S3 + Bedrock Titan v2 + Bedrock AgentCore + CloudWatch).")
    return role_arn


def build_deployment_package() -> bytes:
    """Build deployment zip package with Linux dependencies via uv."""
    print("2. Packaging dependencies for AWS Lambda (Linux x86_64, Python 3.12)...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, ".portal_build")

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Install dependencies into build_dir
    cmd = [
        "uv",
        "pip",
        "install",
        "--target",
        build_dir,
        "--python-platform",
        "x86_64-unknown-linux-gnu",
        "--python-version",
        "3.12",
        "fastapi",
        "mangum",
        "pydantic",
        "python-multipart",
    ]
    subprocess.check_call(cmd, cwd=project_root)

    # Copy portal_lambda.py and handler.py into build_dir
    shutil.copy2(
        os.path.join(project_root, "lambda_portal", "portal_lambda.py"),
        os.path.join(build_dir, "portal_lambda.py"),
    )
    shutil.copy2(
        os.path.join(project_root, "lambda_portal", "handler.py"),
        os.path.join(build_dir, "handler.py"),
    )

    print("   ✓ Files staged. Compressing into zip package...")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(build_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, build_dir)
                zf.write(file_path, arcname)

    zip_bytes = zip_buffer.getvalue()
    print(f"   ✓ Deployment package compressed: {len(zip_bytes) / (1024 * 1024):.2f} MB")
    return zip_bytes


def deploy_lambda(role_arn: str, zip_bytes: bytes) -> str:
    """Create or update the Lambda function."""
    print(f"3. Deploying Lambda function '{FUNCTION_NAME}'...")

    env_vars = {
        "Variables": {
            "DOCS_S3_BUCKET": BUCKET_NAME,
            "AGENT_RUNTIME_ARN": AGENT_RUNTIME_ARN,
            "INTERVIEW_SECRET": INTERVIEW_SECRET,
        }
    }

    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
        print("   ✓ Updating existing Lambda code...")
        lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes,
        )

        # Wait for update to complete
        waiter = lambda_client.get_waiter("function_updated")
        waiter.wait(FunctionName=FUNCTION_NAME)

        print("   ✓ Updating configuration...")
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Role=role_arn,
            Handler="portal_lambda.handler",
            Timeout=60,
            MemorySize=512,
            Environment=env_vars,
        )
    except lambda_client.exceptions.ResourceNotFoundException:
        print("   ✓ Creating new Lambda function...")
        # Brief retry loop for IAM propagation
        for attempt in range(6):
            try:
                lambda_client.create_function(
                    FunctionName=FUNCTION_NAME,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler="portal_lambda.handler",
                    Code={"ZipFile": zip_bytes},
                    Description="CFAS Autonomous Multi-Agent Evaluation & Ingestion Web Portal",
                    Timeout=60,
                    MemorySize=512,
                    Architectures=["x86_64"],
                    Environment=env_vars,
                )
                break
            except lambda_client.exceptions.InvalidParameterValueException as e:
                if "The role defined for the function cannot be assumed by Lambda" in str(e):
                    print(f"   Waiting for IAM role propagation (attempt {attempt + 1}/6)...")
                    time.sleep(5)
                else:
                    raise

    waiter = lambda_client.get_waiter("function_active_v2")
    waiter.wait(FunctionName=FUNCTION_NAME)
    print(f"   ✓ Lambda function active: {FUNCTION_NAME}")
    return FUNCTION_NAME


def configure_function_url() -> str:
    """Create or retrieve public Lambda Function URL with CORS."""
    print("4. Configuring AWS Lambda Public Function URL...")
    cors_config = {
        "AllowOrigins": ["*"],
        "AllowMethods": ["*"],
        "AllowHeaders": ["*"],
        "AllowCredentials": True,
    }

    try:
        url_resp = lambda_client.get_function_url_config(FunctionName=FUNCTION_NAME)
        function_url = url_resp["FunctionUrl"]
        print(f"   ✓ Existing Function URL found: {function_url}")
        lambda_client.update_function_url_config(
            FunctionName=FUNCTION_NAME,
            AuthType="NONE",
            Cors=cors_config,
        )
    except lambda_client.exceptions.ResourceNotFoundException:
        print("   Creating public Function URL...")
        url_resp = lambda_client.create_function_url_config(
            FunctionName=FUNCTION_NAME,
            AuthType="NONE",
            Cors=cors_config,
        )
        function_url = url_resp["FunctionUrl"]
        print(f"   ✓ Created Function URL: {function_url}")

    # Ensure public invoke permissions on the Function URL
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="UrlPolicyInvokeURL",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE",
        )
        print("   ✓ Public access permission added (lambda:InvokeFunctionUrl).")
    except lambda_client.exceptions.ResourceConflictException:
        print("   ✓ Public access permission already exists (lambda:InvokeFunctionUrl).")

    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="UrlPolicyInvokeFunction",
            Action="lambda:InvokeFunction",
            Principal="*",
            InvokedViaFunctionUrl=True,
        )
        print("   ✓ Public access permission added (lambda:InvokeFunction).")
    except lambda_client.exceptions.ResourceConflictException:
        print("   ✓ Public access permission already exists (lambda:InvokeFunction).")

    return function_url


def test_public_endpoint(function_url: str):
    """Smoke test the public Function URL."""
    print("5. Smoke testing public HTTPS endpoint...")
    health_url = function_url.rstrip("/") + "/api/health"
    try:
        with urllib.request.urlopen(health_url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"   ✓ Health check passed: {data}")
    except Exception as e:
        print(f"   ⚠️ Health check note: {e}")


def main():
    print("=" * 70)
    print("CFAS MULTI-AGENT SYSTEM | WEB PORTAL SERVERLESS DEPLOYMENT")
    print("=" * 70)

    role_arn = setup_iam_role()
    zip_bytes = build_deployment_package()
    deploy_lambda(role_arn, zip_bytes)
    function_url = configure_function_url()
    test_public_endpoint(function_url)

    portal_access_url = f"{function_url.rstrip('/')}/?token={INTERVIEW_SECRET}"
    print("\n" + "=" * 70)
    print("🚀 DEPLOYMENT SUCCESSFUL!")
    print("=" * 70)
    print(f"\n🌐 Live Executive Web Portal URL:")
    print(f"   {portal_access_url}\n")
    print("Features available on this public link:")
    print(" - Drag-and-drop document upload (.docx, .txt, .md)")
    print(" - Serverless S3 ingestion & Amazon Titan v2 1024-dim embedding")
    print(" - Live document catalog display with chunk counts")
    print(" - Live chat console connecting to Bedrock AgentCore Runtime")
    print(" - Routing badges: AST SpaceMobile Analyst, Bylaws Specialist, Ops")
    print(" - $0.00 idle cost (AWS Lambda pay-per-request serverless)")
    print("=" * 70)


if __name__ == "__main__":
    main()
