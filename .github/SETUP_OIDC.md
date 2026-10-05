# 🔐 Setting Up AWS OIDC for GitHub Actions CI/CD

This project uses **OpenID Connect (OIDC)** to authenticate GitHub Actions with AWS. 

> [!IMPORTANT]
> **Enterprise Best Practice:** Never store static `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` in GitHub Secrets. OIDC exchanges short-lived cryptographic tokens directly with AWS IAM, eliminating long-term credential leakage risks.

---

### Step 1: Create the GitHub OIDC Identity Provider in AWS (Run Once per AWS Account)

In the AWS Console (IAM ➔ Identity Providers) or via AWS CLI:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  --thumbprint-list 1c58a3a8518e8759bf075b76b750d4f8d264fcd9
```

---

### Step 2: Create the IAM Role for GitHub Actions

Create an IAM Role named `GitHubActionsAgentCoreDeployRole`:

#### 1. Trust Relationship Policy
Configure the trust relationship for your repository (`jr192/BedrockAgentCoreApp`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::725079717969:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:jr192/BedrockAgentCoreApp:*"
        }
      }
    }
  ]
}
```

#### 2. Permissions Policies
Attach the following managed or custom policies to the role:
- `AdministratorAccess` (or scoped permissions for CloudFormation, Lambda, ECR, Bedrock, S3, and IAM).

---

### Step 3: Add the Secret to Your GitHub Repository

1. In your GitHub repository, navigate to **Settings** ➔ **Secrets and variables** ➔ **Actions**.
2. Click **New repository secret**.
3. Name: `AWS_ROLE_ARN`
4. Value: `arn:aws:iam::725079717969:role/GitHubActionsAgentCoreDeployRole`

---

### Step 4: Push to Main to Trigger Deployment

Once configured, any push to `main` will:
1. Run quality, linting, and session isolation tests.
2. Run the **LLMOps Evaluation Gate** on 10 Golden Benchmark test cases.
3. Package and deploy the containerized runtime to AWS Bedrock AgentCore.
4. Execute an automated cloud smoke test to verify live response fidelity.
