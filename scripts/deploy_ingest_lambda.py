"""
Automated Deployment Script for S3 Event-Driven Ingestion Lambda
Deploys:
1. Least-privilege IAM Execution Role for Lambda (S3 + Bedrock Titan v2 + CloudWatch).
2. Cloud-native Lambda function `corporate-bylaws-ingest-worker`.
3. S3 Bucket Event Notification trigger on `uploads/`.
"""

import io
import json
import os
import sys
import time
import zipfile
import boto3

REGION = "us-east-1"
BUCKET_NAME = "cfas-corporate-docs-725079717969"
FUNCTION_NAME = "corporate-bylaws-ingest-worker"
ROLE_NAME = "CorporateBylawsIngestLambdaRole"

iam = boto3.client("iam", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
s3_client = boto3.client("s3", region_name=REGION)


def setup_iam_role() -> str:
    """Create or get the IAM execution role with necessary least-privilege permissions."""
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
            Description="Execution role for Corporate Bylaws S3 Ingest Lambda",
        )
        role_arn = role["Role"]["Arn"]
        print(f"   ✓ Created IAM role: {role_arn}")

    # Attach least-privilege inline policy
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
                ],
            },
        ],
    }

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="CorporateBylawsIngestPolicy",
        PolicyDocument=json.dumps(policy_doc),
    )
    print("   ✓ Inline policy attached.")
    return role_arn


def create_deployment_package() -> bytes:
    """Create in-memory zip archive with lambda_ingest/handler.py."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    handler_path = os.path.join(root_dir, "lambda_ingest", "handler.py")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        # Store as handler.py in root of zip
        z.write(handler_path, arcname="handler.py")

    zip_bytes = zip_buffer.getvalue()
    print(f"   ✓ Deployment package built ({len(zip_bytes) / 1024:.1f} KB).")
    return zip_bytes


def deploy_lambda_function(role_arn: str, zip_bytes: bytes) -> str:
    """Create or update the Lambda function in AWS."""
    print(f"3. Deploying Lambda function: {FUNCTION_NAME}...")

    # Wait briefly for IAM eventual consistency if newly created
    time.sleep(5)

    try:
        fn = lambda_client.get_function(FunctionName=FUNCTION_NAME)
        print("   Updating existing Lambda function code and configuration...")
        lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Role=role_arn,
            Handler="handler.lambda_handler",
            Runtime="python3.12",
            Timeout=300,
            MemorySize=512,
            Environment={
                "Variables": {
                    "DEFAULT_BUCKET": BUCKET_NAME,
                }
            },
        )
        fn_arn = fn["Configuration"]["FunctionArn"]
        print(f"   ✓ Updated Lambda: {fn_arn}")
    except lambda_client.exceptions.ResourceNotFoundException:
        print("   Creating new Lambda function...")
        # Retry loop for IAM role propagation
        for attempt in range(6):
            try:
                fn = lambda_client.create_function(
                    FunctionName=FUNCTION_NAME,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler="handler.lambda_handler",
                    Code={"ZipFile": zip_bytes},
                    Description="S3 event-driven semantic chunking and embedding worker",
                    Timeout=300,
                    MemorySize=512,
                    Architectures=["arm64"],
                    Environment={
                        "Variables": {
                            "DEFAULT_BUCKET": BUCKET_NAME,
                        }
                    },
                )
                fn_arn = fn["FunctionArn"]
                print(f"   ✓ Created Lambda: {fn_arn}")
                break
            except lambda_client.exceptions.InvalidParameterValueException as e:
                if "role" in str(e).lower() and attempt < 5:
                    print(f"   Waiting for IAM role propagation (attempt {attempt+1}/6)...")
                    time.sleep(5)
                else:
                    raise

    return fn_arn


def configure_s3_trigger(fn_arn: str):
    """Grant S3 permission to invoke Lambda and configure ObjectCreated notification."""
    print(f"4. Configuring S3 Event Notification on {BUCKET_NAME}...")

    # 1. Add Lambda permission for S3 invocation
    statement_id = "s3-upload-trigger-permission"
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{BUCKET_NAME}",
        )
        print("   ✓ Lambda invoke permission granted to S3.")
    except lambda_client.exceptions.ResourceConflictException:
        print("   ✓ Lambda permission already exists.")

    # 2. Attach S3 bucket notification for prefix uploads/
    notification_config = {
        "LambdaFunctionConfigurations": [
            {
                "Id": "BylawsDocumentUploadTrigger",
                "LambdaFunctionArn": fn_arn,
                "Events": ["s3:ObjectCreated:*"],
                "Filter": {
                    "Key": {
                        "FilterRules": [
                            {"Name": "prefix", "Value": "uploads/"}
                        ]
                    }
                },
            }
        ]
    }

    s3_client.put_bucket_notification_configuration(
        Bucket=BUCKET_NAME,
        NotificationConfiguration=notification_config,
    )
    print(f"   ✓ S3 Event Trigger active on s3://{BUCKET_NAME}/uploads/*")


def main():
    print("=" * 70)
    print("🚀 AUTOMATED DEPLOYMENT: S3 EVENT-DRIVEN INGESTION LAMBDA")
    print("=" * 70)
    role_arn = setup_iam_role()
    zip_bytes = create_deployment_package()
    fn_arn = deploy_lambda_function(role_arn, zip_bytes)
    configure_s3_trigger(fn_arn)
    print("=" * 70)
    print("🎉 DEPLOYMENT COMPLETE!")
    print(f"Function ARN : {fn_arn}")
    print(f"S3 Trigger   : s3://{BUCKET_NAME}/uploads/ ➔ {FUNCTION_NAME}")
    print("=" * 70)


if __name__ == "__main__":
    main()
