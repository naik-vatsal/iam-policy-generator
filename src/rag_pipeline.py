"""
RAG-Enhanced IAM Policy Generator
==================================
Uses Retrieval-Augmented Generation to ground the model's output
in real AWS IAM actions, reducing hallucinated action names.

Architecture:
1. User provides natural language description
2. We detect which AWS services are mentioned
3. We retrieve RELEVANT actions (not all) for those services
4. We inject the actions into the prompt as context
5. The fine-tuned model generates the policy with grounded actions

Usage:
    from rag_pipeline import RAGPolicyGenerator
    generator = RAGPolicyGenerator("results/config_a/final_model")
    result = generator.generate("Allow read-only access to S3 bucket my-data")
    print(result["policy"])
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# AWS ACTION CATALOG (20+ services, 300+ actions)
# ============================================================

AWS_ACTION_CATALOG = {
    "s3": {
        "display_name": "Amazon S3",
        "read_actions": [
            "s3:GetObject", "s3:GetObjectVersion", "s3:GetObjectAcl",
            "s3:GetBucketLocation", "s3:GetBucketPolicy", "s3:GetBucketAcl",
            "s3:GetBucketVersioning", "s3:GetEncryptionConfiguration"
        ],
        "list_actions": [
            "s3:ListBucket", "s3:ListAllMyBuckets", "s3:ListBucketVersions",
            "s3:ListBucketMultipartUploads"
        ],
        "write_actions": [
            "s3:PutObject", "s3:PutObjectAcl", "s3:PutBucketPolicy",
            "s3:PutEncryptionConfiguration", "s3:PutLifecycleConfiguration",
            "s3:PutBucketTagging", "s3:PutBucketVersioning"
        ],
        "delete_actions": [
            "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:DeleteBucket",
            "s3:DeleteBucketPolicy"
        ],
        "wildcard": "s3:*"
    },
    "ec2": {
        "display_name": "Amazon EC2",
        "read_actions": [
            "ec2:DescribeInstances", "ec2:DescribeInstanceStatus",
            "ec2:DescribeSecurityGroups", "ec2:DescribeVpcs",
            "ec2:DescribeSubnets", "ec2:DescribeImages",
            "ec2:DescribeVolumes", "ec2:DescribeSnapshots",
            "ec2:DescribeKeyPairs", "ec2:DescribeRegions",
            "ec2:DescribeAvailabilityZones"
        ],
        "manage_actions": [
            "ec2:RunInstances", "ec2:StartInstances", "ec2:StopInstances",
            "ec2:RebootInstances", "ec2:TerminateInstances",
            "ec2:CreateTags", "ec2:DeleteTags"
        ],
        "security_actions": [
            "ec2:AuthorizeSecurityGroupIngress", "ec2:AuthorizeSecurityGroupEgress",
            "ec2:RevokeSecurityGroupIngress", "ec2:RevokeSecurityGroupEgress",
            "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup"
        ],
        "volume_actions": [
            "ec2:CreateVolume", "ec2:DeleteVolume", "ec2:AttachVolume",
            "ec2:DetachVolume", "ec2:CreateSnapshot", "ec2:DeleteSnapshot"
        ],
        "wildcard": "ec2:*"
    },
    "lambda": {
        "display_name": "AWS Lambda",
        "read_actions": [
            "lambda:GetFunction", "lambda:GetFunctionConfiguration",
            "lambda:ListFunctions", "lambda:ListAliases", "lambda:ListTags"
        ],
        "write_actions": [
            "lambda:CreateFunction", "lambda:UpdateFunctionCode",
            "lambda:UpdateFunctionConfiguration", "lambda:PublishVersion",
            "lambda:CreateAlias", "lambda:UpdateAlias", "lambda:TagResource"
        ],
        "invoke_actions": ["lambda:InvokeFunction", "lambda:InvokeAsync"],
        "delete_actions": ["lambda:DeleteFunction", "lambda:DeleteAlias"],
        "event_actions": [
            "lambda:CreateEventSourceMapping", "lambda:DeleteEventSourceMapping",
            "lambda:GetEventSourceMapping", "lambda:ListEventSourceMappings",
            "lambda:UpdateEventSourceMapping"
        ],
        "wildcard": "lambda:*"
    },
    "dynamodb": {
        "display_name": "Amazon DynamoDB",
        "read_actions": [
            "dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:Query",
            "dynamodb:Scan", "dynamodb:DescribeTable"
        ],
        "write_actions": [
            "dynamodb:PutItem", "dynamodb:UpdateItem",
            "dynamodb:BatchWriteItem", "dynamodb:DeleteItem"
        ],
        "admin_actions": [
            "dynamodb:CreateTable", "dynamodb:DeleteTable", "dynamodb:UpdateTable",
            "dynamodb:CreateBackup", "dynamodb:DeleteBackup",
            "dynamodb:RestoreTableFromBackup", "dynamodb:ListTables"
        ],
        "stream_actions": [
            "dynamodb:GetRecords", "dynamodb:GetShardIterator",
            "dynamodb:DescribeStream", "dynamodb:ListStreams"
        ],
        "wildcard": "dynamodb:*"
    },
    "logs": {
        "display_name": "Amazon CloudWatch Logs",
        "read_actions": [
            "logs:GetLogEvents", "logs:FilterLogEvents",
            "logs:DescribeLogGroups", "logs:DescribeLogStreams"
        ],
        "write_actions": [
            "logs:CreateLogGroup", "logs:CreateLogStream",
            "logs:PutLogEvents", "logs:PutRetentionPolicy"
        ],
        "delete_actions": ["logs:DeleteLogGroup", "logs:DeleteLogStream"],
        "wildcard": "logs:*"
    },
    "cloudwatch": {
        "display_name": "Amazon CloudWatch",
        "read_actions": [
            "cloudwatch:GetMetricData", "cloudwatch:GetMetricStatistics",
            "cloudwatch:ListMetrics", "cloudwatch:DescribeAlarms",
            "cloudwatch:GetDashboard", "cloudwatch:ListDashboards"
        ],
        "write_actions": [
            "cloudwatch:PutMetricData", "cloudwatch:PutMetricAlarm",
            "cloudwatch:DeleteAlarms", "cloudwatch:PutDashboard"
        ],
        "wildcard": "cloudwatch:*"
    },
    "iam": {
        "display_name": "AWS IAM",
        "read_actions": [
            "iam:GetUser", "iam:GetRole", "iam:GetPolicy", "iam:GetGroup",
            "iam:ListUsers", "iam:ListRoles", "iam:ListPolicies", "iam:ListGroups",
            "iam:ListAttachedRolePolicies", "iam:ListAttachedUserPolicies"
        ],
        "user_self_actions": [
            "iam:ChangePassword", "iam:CreateAccessKey", "iam:DeleteAccessKey",
            "iam:UpdateAccessKey", "iam:ListAccessKeys",
            "iam:CreateVirtualMFADevice", "iam:EnableMFADevice",
            "iam:DeactivateMFADevice", "iam:ListMFADevices"
        ],
        "admin_actions": [
            "iam:CreateRole", "iam:DeleteRole", "iam:PassRole",
            "iam:AttachRolePolicy", "iam:DetachRolePolicy",
            "iam:PutRolePolicy", "iam:DeleteRolePolicy",
            "iam:CreateInstanceProfile", "iam:DeleteInstanceProfile"
        ],
        "wildcard": "iam:*"
    },
    "sts": {
        "display_name": "AWS STS",
        "actions": [
            "sts:AssumeRole", "sts:AssumeRoleWithSAML",
            "sts:AssumeRoleWithWebIdentity", "sts:GetSessionToken",
            "sts:GetCallerIdentity"
        ]
    },
    "kms": {
        "display_name": "AWS KMS",
        "crypto_actions": [
            "kms:Encrypt", "kms:Decrypt", "kms:ReEncryptFrom", "kms:ReEncryptTo",
            "kms:GenerateDataKey", "kms:GenerateDataKeyWithoutPlaintext", "kms:DescribeKey"
        ],
        "admin_actions": [
            "kms:CreateKey", "kms:CreateAlias", "kms:DeleteAlias",
            "kms:DisableKey", "kms:EnableKey", "kms:ScheduleKeyDeletion",
            "kms:ListKeys", "kms:ListAliases"
        ],
        "wildcard": "kms:*"
    },
    "sns": {
        "display_name": "Amazon SNS",
        "actions": [
            "sns:Publish", "sns:Subscribe", "sns:Unsubscribe",
            "sns:CreateTopic", "sns:DeleteTopic", "sns:ListTopics",
            "sns:GetTopicAttributes", "sns:SetTopicAttributes"
        ],
        "wildcard": "sns:*"
    },
    "sqs": {
        "display_name": "Amazon SQS",
        "actions": [
            "sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage",
            "sqs:GetQueueAttributes", "sqs:GetQueueUrl", "sqs:ListQueues",
            "sqs:CreateQueue", "sqs:DeleteQueue", "sqs:PurgeQueue"
        ],
        "wildcard": "sqs:*"
    },
    "ses": {
        "display_name": "Amazon SES",
        "actions": [
            "ses:SendEmail", "ses:SendRawEmail", "ses:SendTemplatedEmail",
            "ses:GetSendQuota", "ses:VerifyEmailIdentity"
        ]
    },
    "secretsmanager": {
        "display_name": "AWS Secrets Manager",
        "actions": [
            "secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret",
            "secretsmanager:ListSecrets", "secretsmanager:CreateSecret",
            "secretsmanager:UpdateSecret", "secretsmanager:DeleteSecret"
        ]
    },
    "ssm": {
        "display_name": "AWS Systems Manager",
        "actions": [
            "ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath",
            "ssm:PutParameter", "ssm:DeleteParameter", "ssm:DescribeParameters"
        ]
    },
    "ecr": {
        "display_name": "Amazon ECR",
        "actions": [
            "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
            "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
            "ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
            "ecr:CompleteLayerUpload", "ecr:DescribeRepositories"
        ]
    },
    "sagemaker": {
        "display_name": "Amazon SageMaker",
        "actions": [
            "sagemaker:CreateNotebookInstance", "sagemaker:StartNotebookInstance",
            "sagemaker:StopNotebookInstance", "sagemaker:CreateTrainingJob",
            "sagemaker:DescribeTrainingJob", "sagemaker:ListTrainingJobs",
            "sagemaker:CreateModel", "sagemaker:InvokeEndpoint"
        ]
    },
    "cloudtrail": {
        "display_name": "AWS CloudTrail",
        "actions": [
            "cloudtrail:DescribeTrails", "cloudtrail:GetTrailStatus",
            "cloudtrail:LookupEvents", "cloudtrail:CreateTrail",
            "cloudtrail:StartLogging", "cloudtrail:StopLogging",
            "cloudtrail:DeleteTrail", "cloudtrail:UpdateTrail"
        ]
    },
    "rds": {
        "display_name": "Amazon RDS",
        "actions": [
            "rds:DescribeDBInstances", "rds:CreateDBInstance",
            "rds:DeleteDBInstance", "rds:StartDBInstance", "rds:StopDBInstance",
            "rds:RebootDBInstance", "rds:CreateDBSnapshot", "rds:RestoreDBInstanceFromDBSnapshot"
        ]
    },
    "route53": {
        "display_name": "Amazon Route 53",
        "actions": [
            "route53:ChangeResourceRecordSets", "route53:GetHostedZone",
            "route53:ListHostedZones", "route53:ListResourceRecordSets",
            "route53:CreateHostedZone", "route53:DeleteHostedZone"
        ]
    },
    "cloudformation": {
        "display_name": "AWS CloudFormation",
        "actions": [
            "cloudformation:CreateStack", "cloudformation:UpdateStack",
            "cloudformation:DeleteStack", "cloudformation:DescribeStacks",
            "cloudformation:DescribeStackEvents", "cloudformation:GetTemplate",
            "cloudformation:ListStacks"
        ]
    },
    "elasticloadbalancing": {
        "display_name": "Elastic Load Balancing",
        "actions": [
            "elasticloadbalancing:CreateLoadBalancer", "elasticloadbalancing:DeleteLoadBalancer",
            "elasticloadbalancing:DescribeLoadBalancers",
            "elasticloadbalancing:CreateTargetGroup", "elasticloadbalancing:RegisterTargets",
            "elasticloadbalancing:CreateListener", "elasticloadbalancing:DescribeTargetHealth"
        ]
    }
}

# ============================================================
# SERVICE DETECTION (precise aliases to avoid false positives)
# ============================================================

SERVICE_ALIASES = {
    "s3": ["s3 bucket", "s3 object", "s3 read", "s3 write", "s3 access", " s3 ", "from s3", "to s3", "simple storage"],
    "ec2": ["ec2 instance", "ec2 ", "virtual machine", "compute instance"],
    "lambda": ["lambda function", "lambda "],
    "dynamodb": ["dynamodb", "dynamo db"],
    "logs": ["cloudwatch log", "log group", "log stream", "write logs", "cloudwatch logs"],
    "cloudwatch": ["cloudwatch metric", "cloudwatch alarm", "cloudwatch dashboard", "monitoring metric"],
    "iam": ["iam user", "iam role", "iam policy", "iam group", "mfa device", "access key", "own password"],
    "sts": ["assume role", "cross-account", "session token"],
    "kms": ["kms key", "encrypt", "decrypt", "key management"],
    "sns": ["sns topic", "sns publish", "notification topic"],
    "sqs": ["sqs queue", "message queue", "dead letter queue"],
    "ses": ["ses ", "send email", "email service"],
    "secretsmanager": ["secrets manager", "secret value", "secretsmanager"],
    "ssm": ["parameter store", "systems manager", "ssm parameter"],
    "ecr": ["container registry", "ecr ", "docker image", "pull image"],
    "sagemaker": ["sagemaker", "notebook instance", "training job"],
    "cloudtrail": ["cloudtrail", "audit trail"],
    "rds": ["rds ", "db instance", "database instance"],
    "route53": ["route53", "route 53", "dns record", "hosted zone"],
    "cloudformation": ["cloudformation", "cfn stack", "cloud formation stack"],
    "elasticloadbalancing": ["load balancer", "target group", "alb ", "nlb "],
}


def detect_services(description):
    """Detect which AWS services are mentioned in the description."""
    desc_lower = " " + description.lower() + " "
    detected = set()
    for service, aliases in SERVICE_ALIASES.items():
        for alias in aliases:
            if alias in desc_lower:
                detected.add(service)
                break
    return detected


def get_relevant_actions(services, description):
    """Retrieve only the RELEVANT actions based on the description intent.
    
    Instead of dumping all actions for a service, we analyze the description
    to determine what kind of access is needed (read, write, delete, etc.)
    and only inject those specific actions. This keeps the context small
    and focused, improving generation quality.
    """
    desc_lower = description.lower()
    context_parts = []

    for service in services:
        if service not in AWS_ACTION_CATALOG:
            continue
        catalog = AWS_ACTION_CATALOG[service]
        display_name = catalog.get("display_name", service)
        selected_actions = []

        # Full access — just use wildcard
        if any(w in desc_lower for w in ["full access", "full ", "admin", " all "]):
            selected_actions = [catalog.get("wildcard", f"{service}:*")]

        else:
            # Read access
            if any(w in desc_lower for w in ["read", "get", "describe", "list", "view", "read-only", "monitor"]):
                for key in ["read_actions", "list_actions"]:
                    if key in catalog:
                        selected_actions.extend(catalog[key][:5])

            # Write access
            if any(w in desc_lower for w in ["write", "put", "create", "update", "upload", "send", "publish", "log"]):
                for key in ["write_actions", "invoke_actions", "actions"]:
                    if key in catalog:
                        selected_actions.extend(catalog[key][:5])

            # Delete access
            if any(w in desc_lower for w in ["delete", "remove", "terminate"]):
                for key in ["delete_actions", "manage_actions"]:
                    if key in catalog:
                        selected_actions.extend(catalog[key][:5])

            # Invoke / execute
            if any(w in desc_lower for w in ["invoke", "call", "trigger", "execute", "run"]):
                for key in ["invoke_actions", "manage_actions", "actions"]:
                    if key in catalog:
                        selected_actions.extend(catalog[key][:5])

            # Deny / restrict
            if any(w in desc_lower for w in ["deny", "prevent", "block", "restrict"]):
                for key in ["delete_actions", "manage_actions", "admin_actions", "actions"]:
                    if key in catalog:
                        selected_actions.extend(catalog[key][:5])

            # Assume role (STS specific)
            if any(w in desc_lower for w in ["assume", "cross-account"]):
                if "actions" in catalog:
                    selected_actions.extend(catalog["actions"][:5])

            # Encrypt / decrypt (KMS specific)
            if any(w in desc_lower for w in ["encrypt", "decrypt", "kms"]):
                for key in ["crypto_actions", "actions"]:
                    if key in catalog:
                        selected_actions.extend(catalog[key][:5])

            # Fallback: if no intent matched, pick top actions
            if not selected_actions:
                for key, value in catalog.items():
                    if key in ("display_name", "wildcard"):
                        continue
                    if isinstance(value, list):
                        selected_actions.extend(value[:3])
                    if len(selected_actions) >= 8:
                        break

        # Deduplicate and limit to 10 per service
        selected_actions = list(dict.fromkeys(selected_actions))[:10]

        if selected_actions:
            context_parts.append(f"{display_name}: {', '.join(selected_actions)}")

    return "\n".join(context_parts)


class RAGPolicyGenerator:
    """RAG-enhanced IAM Policy Generator."""

    def __init__(self, adapter_path, model_name="mistralai/Mistral-7B-v0.3"):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", dtype=torch.bfloat16,
        )

        print("Loading fine-tuned LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("RAG Policy Generator ready!")

    def generate(self, description, max_tokens=512, temperature=0.1, use_rag=True):
        """Generate an IAM policy from a natural language description."""
        # Step 1: Detect services
        detected_services = detect_services(description)

        # Step 2: Retrieve relevant actions (slim, focused)
        rag_context = ""
        if use_rag and detected_services:
            rag_context = get_relevant_actions(detected_services, description)

        # Step 3: Build prompt
        if rag_context:
            prompt = (
                f"### Context:\nRelevant AWS actions:\n{rag_context}\n\n"
                f"### Instruction:\n{description}\n\n### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{description}\n\n### Response:\n"

        # Step 4: Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        policy_text = response.split("### Response:\n")[-1].strip()

        # Step 5: Parse and validate
        result = {
            "raw_output": policy_text,
            "valid_json": False,
            "policy": None,
            "services_detected": sorted(list(detected_services)),
            "rag_context_used": bool(rag_context),
        }
        try:
            result["policy"] = json.loads(policy_text)
            result["valid_json"] = True
        except json.JSONDecodeError:
            pass

        return result

    def generate_and_print(self, description, use_rag=True):
        """Generate and pretty-print a policy."""
        print(f"\nInput: {description}")
        print(f"Services detected: {detect_services(description)}")
        print(f"RAG: {'enabled' if use_rag else 'disabled'}")
        print("-" * 50)

        result = self.generate(description, use_rag=use_rag)
        if result["valid_json"]:
            print(f"✅ Valid JSON!\n{json.dumps(result['policy'], indent=2)}")
        else:
            print(f"❌ Invalid JSON\n{result['raw_output'][:500]}")
        return result

    def compare_rag_vs_no_rag(self, description):
        """Compare output with and without RAG."""
        print("=" * 60)
        print(f"DESCRIPTION: {description}")
        print("=" * 60)

        print("\n--- WITHOUT RAG ---")
        r1 = self.generate(description, use_rag=False)
        if r1["valid_json"]:
            print(f"✅ Valid | Services: {self._extract_services(r1['policy'])}")
            print(json.dumps(r1["policy"], indent=2))
        else:
            print(f"❌ Invalid JSON\n{r1['raw_output'][:300]}")

        print("\n--- WITH RAG ---")
        r2 = self.generate(description, use_rag=True)
        if r2["valid_json"]:
            print(f"✅ Valid | Services: {self._extract_services(r2['policy'])}")
            print(json.dumps(r2["policy"], indent=2))
        else:
            print(f"❌ Invalid JSON\n{r2['raw_output'][:300]}")

        return r1, r2

    def _extract_services(self, policy):
        services = set()
        statements = policy.get("Statement", [])
        if isinstance(statements, dict):
            statements = [statements]
        for stmt in statements:
            for key in ["Action", "NotAction"]:
                actions = stmt.get(key, [])
                if isinstance(actions, str):
                    actions = [actions]
                for action in actions:
                    if ":" in action:
                        services.add(action.split(":")[0])
        return sorted(list(services))


if __name__ == "__main__":
    generator = RAGPolicyGenerator("results/config_a/final_model")
    generator.generate_and_print("Allow read-only access to S3 bucket named customer-data")
    generator.generate_and_print("Allow a Lambda function to read from DynamoDB table users and write logs to CloudWatch")
    generator.generate_and_print("Deny all S3 delete operations across all buckets")
    generator.compare_rag_vs_no_rag("Allow an ECS task to pull images from ECR, read secrets from Secrets Manager, and write logs")