// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Framework/Commands/Commands.h"
#include "VLMDataExporterStyle.h"

class FVLMDataExporterCommands : public TCommands<FVLMDataExporterCommands>
{
public:

	FVLMDataExporterCommands()
		: TCommands<FVLMDataExporterCommands>(TEXT("VLMDataExporter"), NSLOCTEXT("Contexts", "VLMDataExporter", "VLMDataExporter Plugin"), NAME_None, FVLMDataExporterStyle::GetStyleSetName())
	{
	}

	// TCommands<> interface
	virtual void RegisterCommands() override;

public:
	TSharedPtr< FUICommandInfo > PluginAction;
};
