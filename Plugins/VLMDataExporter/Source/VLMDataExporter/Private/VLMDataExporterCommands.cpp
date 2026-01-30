// Copyright Epic Games, Inc. All Rights Reserved.

#include "VLMDataExporterCommands.h"

#define LOCTEXT_NAMESPACE "FVLMDataExporterModule"

void FVLMDataExporterCommands::RegisterCommands()
{
	UI_COMMAND(PluginAction, "[VLMDataExporter] Export", "Export VLM Data", EUserInterfaceActionType::Button, FInputGesture());
}

#undef LOCTEXT_NAMESPACE
