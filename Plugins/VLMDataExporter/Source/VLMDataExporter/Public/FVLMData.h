#pragma once

#include "CoreMinimal.h"
#include "FVLMData.generated.h"

USTRUCT()
struct FVLMData
{
	GENERATED_BODY()

	UPROPERTY()
	FString LevelName;

	UPROPERTY()
	int32 BrickSize;

	UPROPERTY()
	FIntVector IndirectionTextureDimensions;

	UPROPERTY()
	TArray<uint8> IndirectionTextureData;

	UPROPERTY()
	int32 IndirectionTextureDataSize;

	UPROPERTY()
	FIntVector BrickDataDimensions;

	UPROPERTY()
	TArray<uint8> AmbientVectorData;

	UPROPERTY()
	int32 AmbientVectorDataSize;

	UPROPERTY()
	TArray<uint8> SHCoefficient0Data;
	
	UPROPERTY()
	int32 SHCoefficient0DataSize;

	UPROPERTY()
	TArray<uint8> SHCoefficient1Data;
	
	UPROPERTY()
	int32 SHCoefficient1DataSize;

	UPROPERTY()
	TArray<uint8> SHCoefficient2Data;
	
	UPROPERTY()
	int32 SHCoefficient2DataSize;

	UPROPERTY()
	TArray<uint8> SHCoefficient3Data;
	
	UPROPERTY()
	int32 SHCoefficient3DataSize;

	UPROPERTY()
	TArray<uint8> SHCoefficient4Data;
	
	UPROPERTY()
	int32 SHCoefficient4DataSize;

	UPROPERTY()
	TArray<uint8> SHCoefficient5Data;
	
	UPROPERTY()
	int32 SHCoefficient5DataSize;
};