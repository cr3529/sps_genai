"""Validated PNG/JSON endpoints for CIFAR-10 diffusion and energy models."""

from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.cifar_generator import CifarGenerator

router = APIRouter(tags=["Assignment 4 image generators"])


@lru_cache(maxsize=2)
def generator_for(kind):
    try:
        return CifarGenerator(kind)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def pixels(kind, num_samples, steps, seed):
    images = generator_for(kind).generate(num_samples, steps, seed)
    return {"model": kind, "num_samples": num_samples, "shape": [3, 32, 32],
            "pixel_range": [0, 1], "steps": steps, "seed": seed,
            "images": images.tolist()}


@router.get("/diffusion/generate", response_class=Response,
            responses={200: {"content": {"image/png": {}}}})
def diffusion_png(num_samples: int = Query(4, ge=1, le=16),
                  steps: int = Query(100, ge=2, le=200),
                  seed: int | None = Query(None, ge=0, le=2**63-1)):
    """Generate RGB CIFAR-10 images from Gaussian noise using reverse diffusion."""
    return Response(generator_for("diffusion").generate_png(num_samples, steps, seed), media_type="image/png")


@router.get("/diffusion/generate/pixels")
def diffusion_pixels(num_samples: int = Query(1, ge=1, le=16),
                     steps: int = Query(100, ge=2, le=200),
                     seed: int | None = Query(None, ge=0, le=2**63-1)):
    return pixels("diffusion", num_samples, steps, seed)


@router.get("/energy/generate", response_class=Response,
            responses={200: {"content": {"image/png": {}}}})
def energy_png(num_samples: int = Query(4, ge=1, le=16),
               steps: int = Query(256, ge=1, le=512),
               seed: int | None = Query(None, ge=0, le=2**63-1)):
    """Generate RGB CIFAR-10 images by lowering input-image energy with Langevin steps."""
    return Response(generator_for("energy").generate_png(num_samples, steps, seed), media_type="image/png")


@router.get("/energy/generate/pixels")
def energy_pixels(num_samples: int = Query(1, ge=1, le=16),
                  steps: int = Query(256, ge=1, le=512),
                  seed: int | None = Query(None, ge=0, le=2**63-1)):
    return pixels("energy", num_samples, steps, seed)


@router.get("/generators/info")
def generator_info():
    """Training metadata from the exact checkpoints used for inference."""
    return {kind: generator_for(kind).metadata for kind in ("energy", "diffusion")}
