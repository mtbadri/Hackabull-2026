"""
services/brain/services/audio.py — Pygame audio playback service.

Handles device selection at startup and in-memory buffer playback.
Never writes audio to disk (Requirement 4.2, 5.1).

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import io
import logging

import pygame

logger = logging.getLogger(__name__)


def init_pygame(device: str) -> None:
    """Initialise the Pygame mixer targeting the given audio device.

    Quits any existing mixer instance first so that pre_init device
    selection is never silently ignored (e.g. when the module was already
    imported by a reloader parent process).

    Attempts to pre-init with the target device name. If that fails,
    falls back to the default system audio device and logs a warning.
    If the fallback also fails, logs an error and returns without raising.

    Requirements: 5.2, 5.3, 5.4
    """
    # Ensure a clean slate — pre_init is a no-op if mixer is already running.
    if pygame.mixer.get_init():
        pygame.mixer.quit()

    try:
        pygame.mixer.pre_init(devicename=device)
        pygame.mixer.init()
        logger.warning("Pygame mixer initialised with device: %s", device)
    except Exception as exc:
        logger.warning(
            "Pygame mixer failed to init with device '%s': %s — falling back to default",
            device,
            exc,
        )
        try:
            pygame.mixer.pre_init()  # reset device name so fallback uses system default
            pygame.mixer.init()
            logger.warning("Pygame mixer initialised with default audio device (fallback)")
        except Exception as fallback_exc:
            logger.error("Pygame mixer fallback init also failed: %s", fallback_exc)


def play_audio(buffer: io.BytesIO) -> None:
    """Play an in-memory audio buffer through the Pygame mixer.

    Blocks until playback is complete. Never raises — all exceptions
    are caught and logged internally.

    Requirements: 5.1, 5.3, 5.5, 5.6
    """
    try:
        pygame.mixer.music.load(buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    except Exception as exc:
        logger.error("Pygame playback failed: %s", exc)
