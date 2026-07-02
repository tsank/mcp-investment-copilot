"""
api/routes/aws.py

AWS ECS control endpoints for the MCP Investment Copilot.

Endpoints:
    GET  /api/v1/aws/status  — returns running/desired count for both services
    POST /api/v1/aws/start   — sets desired-count to 1 for both services
    POST /api/v1/aws/stop    — sets desired-count to 0 for both services

ECS cluster and service names are hardcoded constants — they were created
once and never change for this deployment.
"""

from __future__ import annotations

import boto3
from fastapi import APIRouter, HTTPException

router = APIRouter()

AWS_REGION       = "ap-south-1"
ECS_CLUSTER      = "mcp-copilot"
SERVICES         = ["mcp-copilot-api", "mcp-copilot-ui"]


def _ecs_client():
    return boto3.client("ecs", region_name=AWS_REGION)


@router.get("/status")
async def aws_status():
    """
    Returns running and desired count for both ECS services.
    Status is 'running' only when both services have runningCount == 1.
    """
    try:
        client = _ecs_client()
        response = client.describe_services(
            cluster=ECS_CLUSTER,
            services=SERVICES,
        )
        services = response["services"]
        result = {
            svc["serviceName"]: {
                "running_count": svc["runningCount"],
                "desired_count": svc["desiredCount"],
            }
            for svc in services
        }
        all_running = all(
            v["running_count"] == 1 and v["desired_count"] == 1
            for v in result.values()
        )
        return {
            "status": "running" if all_running else "stopped",
            "services": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def aws_start():
    """Sets desired-count to 1 for both ECS services."""
    try:
        client = _ecs_client()
        for service in SERVICES:
            client.update_service(
                cluster=ECS_CLUSTER,
                service=service,
                desiredCount=1,
            )
        return {"status": "starting", "desired_count": 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def aws_stop():
    """Sets desired-count to 0 for both ECS services."""
    try:
        client = _ecs_client()
        for service in SERVICES:
            client.update_service(
                cluster=ECS_CLUSTER,
                service=service,
                desiredCount=0,
            )
        return {"status": "stopping", "desired_count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))