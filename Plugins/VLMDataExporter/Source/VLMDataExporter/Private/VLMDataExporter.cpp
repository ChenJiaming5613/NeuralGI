// Copyright Epic Games, Inc. All Rights Reserved.

#include "VLMDataExporter.h"

#include "FVLMData.h"
#include "JsonObjectConverter.h"
#include "PrecomputedVolumetricLightmap.h"
#include "VLMDataExporterStyle.h"
#include "VLMDataExporterCommands.h"
#include "Misc/MessageDialog.h"
#include "ToolMenus.h"
#include "Misc/FileHelper.h"

#define LOCTEXT_NAMESPACE "FVLMDataExporterModule"

void FVLMDataExporterModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	
	FVLMDataExporterStyle::Initialize();
	FVLMDataExporterStyle::ReloadTextures();

	FVLMDataExporterCommands::Register();
	
	PluginCommands = MakeShareable(new FUICommandList);

	PluginCommands->MapAction(
		FVLMDataExporterCommands::Get().PluginAction,
		FExecuteAction::CreateRaw(this, &FVLMDataExporterModule::PluginButtonClicked),
		FCanExecuteAction());

	UToolMenus::RegisterStartupCallback(FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FVLMDataExporterModule::RegisterMenus));
}

void FVLMDataExporterModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.

	UToolMenus::UnRegisterStartupCallback(this);

	UToolMenus::UnregisterOwner(this);

	FVLMDataExporterStyle::Shutdown();

	FVLMDataExporterCommands::Unregister();
}

void FVLMDataExporterModule::PluginButtonClicked()
{
	if (!GEditor) return;
	const UWorld* World = GEditor->GetEditorWorldContext().World();
	if (!World) return;
	const ULevel* CurrentLevel = World->GetCurrentLevel();
	if (!CurrentLevel) return;

	const FPrecomputedVolumetricLightmapData* VlmData = CurrentLevel->PrecomputedVolumetricLightmap->Data;
	if (!VlmData)
	{
		UE_LOG(LogTemp, Warning, TEXT("The current level does not have VLM Data!"));
		return;
	}

	FVLMData DataToExport;
	DataToExport.LevelName = FPackageName::GetShortName(World->GetOutermost()->GetName());
	DataToExport.BrickSize = VlmData->BrickSize;
	DataToExport.IndirectionTextureDimensions = VlmData->IndirectionTextureDimensions;
	DataToExport.IndirectionTextureDataSize = VlmData->IndirectionTexture.DataSize;
	DataToExport.IndirectionTextureData = VlmData->IndirectionTexture.Data;
	DataToExport.BrickDataDimensions = VlmData->BrickDataDimensions;
	DataToExport.AmbientVectorData = VlmData->BrickData.AmbientVector.Data;
	DataToExport.AmbientVectorDataSize = VlmData->BrickData.AmbientVector.DataSize;
	DataToExport.SHCoefficient0Data = VlmData->BrickData.SHCoefficients[0].Data;
	DataToExport.SHCoefficient0DataSize = VlmData->BrickData.SHCoefficients[0].DataSize;
	DataToExport.SHCoefficient1Data = VlmData->BrickData.SHCoefficients[1].Data;
	DataToExport.SHCoefficient1DataSize = VlmData->BrickData.SHCoefficients[1].DataSize;
	DataToExport.SHCoefficient2Data = VlmData->BrickData.SHCoefficients[2].Data;
	DataToExport.SHCoefficient2DataSize = VlmData->BrickData.SHCoefficients[2].DataSize;
	DataToExport.SHCoefficient3Data = VlmData->BrickData.SHCoefficients[3].Data;
	DataToExport.SHCoefficient3DataSize = VlmData->BrickData.SHCoefficients[3].DataSize;
	DataToExport.SHCoefficient4Data = VlmData->BrickData.SHCoefficients[4].Data;
	DataToExport.SHCoefficient4DataSize = VlmData->BrickData.SHCoefficients[4].DataSize;
	DataToExport.SHCoefficient5Data = VlmData->BrickData.SHCoefficients[5].Data;
	DataToExport.SHCoefficient5DataSize = VlmData->BrickData.SHCoefficients[5].DataSize;

	FString JsonString;
	if (FJsonObjectConverter::UStructToJsonObjectString(DataToExport, JsonString))
	{
		FString FileName = FString::Printf(TEXT("VLM_%s.json"), *DataToExport.LevelName);
		FString SavePath = FPaths::Combine(FPaths::ProjectSavedDir(), FileName);

		if (FFileHelper::SaveStringToFile(JsonString, *SavePath))
		{
			FString Text = FString::Printf(TEXT("Export successful! File path: %s\nIndirection Texture Dimensions: %d, %d, %d"),
				*SavePath, DataToExport.IndirectionTextureDimensions.X, DataToExport.IndirectionTextureDimensions.Y, DataToExport.IndirectionTextureDimensions.Z);
			FMessageDialog::Open(EAppMsgType::Ok, FText::FromString(Text));
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("Failed to save the file: %s"), *SavePath);
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("JSON serialization failed, please check the data structure!"));
	}
}

void FVLMDataExporterModule::RegisterMenus()
{
	// Owner will be used for cleanup in call to UToolMenus::UnregisterOwner
	FToolMenuOwnerScoped OwnerScoped(this);

	{
		UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Window");
		{
			FToolMenuSection& Section = Menu->FindOrAddSection("WindowLayout");
			Section.AddMenuEntryWithCommandList(FVLMDataExporterCommands::Get().PluginAction, PluginCommands);
		}
	}

	{
		UToolMenu* ToolbarMenu = UToolMenus::Get()->ExtendMenu("LevelEditor.LevelEditorToolBar");
		{
			FToolMenuSection& Section = ToolbarMenu->FindOrAddSection("Settings");
			{
				FToolMenuEntry& Entry = Section.AddEntry(FToolMenuEntry::InitToolBarButton(FVLMDataExporterCommands::Get().PluginAction));
				Entry.SetCommandList(PluginCommands);
			}
		}
	}
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FVLMDataExporterModule, VLMDataExporter)