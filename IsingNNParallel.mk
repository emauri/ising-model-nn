##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=IsingNNParallel
ConfigurationName      :=Debug
WorkspacePath          :=/home/robert/Documents/Primesieve
ProjectPath            :=/home/robert/Documents/IsingNNParallel
IntermediateDirectory  :=./Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=Robert
Date                   :=18/01/17
CodeLitePath           :=/home/robert/.codelite
LinkerName             :=/usr/bin/g++
SharedObjectLinkerName :=/usr/bin/g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="IsingNNParallel.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            := -pthread -lrt -lm -static -static-libgcc -static-libstdc++ -lpthread -llapack -lopenblas 
IncludePath            :=  $(IncludeSwitch)/home/robert/Downloads/OpenBLAS-0.2.19/ $(IncludeSwitch). 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)mcbsp1.2.0 $(LibrarySwitch)openblas 
ArLibs                 :=  "libmcbsp1.2.0.a" "libopenblas.a" 
LibPath                := $(LibraryPathSwitch)/home/robert/MulticoreBSP-for-C/lib/ $(LibraryPathSwitch)/usr/lib $(LibraryPathSwitch)/usr/local/lib $(LibraryPathSwitch)/home/robert/Downloads/armadillo-7.600.2 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/ar rcu
CXX      := /usr/bin/g++
CC       := /usr/bin/gcc
CXXFLAGS := -std=c++11 -O3  -g -O0 -Wall $(Preprocessors)
CFLAGS   :=  -g -O0 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/IsingDataLoader.cpp$(ObjectSuffix) $(IntermediateDirectory)/NetworkTrainer.cpp$(ObjectSuffix) $(IntermediateDirectory)/ShallowNetwork.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@test -d ./Debug || $(MakeDirCommand) ./Debug


$(IntermediateDirectory)/.d:
	@test -d ./Debug || $(MakeDirCommand) ./Debug

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/robert/Documents/IsingNNParallel/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix)main.cpp

$(IntermediateDirectory)/IsingDataLoader.cpp$(ObjectSuffix): IsingDataLoader.cpp $(IntermediateDirectory)/IsingDataLoader.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/robert/Documents/IsingNNParallel/IsingDataLoader.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/IsingDataLoader.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/IsingDataLoader.cpp$(DependSuffix): IsingDataLoader.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/IsingDataLoader.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/IsingDataLoader.cpp$(DependSuffix) -MM IsingDataLoader.cpp

$(IntermediateDirectory)/IsingDataLoader.cpp$(PreprocessSuffix): IsingDataLoader.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/IsingDataLoader.cpp$(PreprocessSuffix)IsingDataLoader.cpp

$(IntermediateDirectory)/NetworkTrainer.cpp$(ObjectSuffix): NetworkTrainer.cpp $(IntermediateDirectory)/NetworkTrainer.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/robert/Documents/IsingNNParallel/NetworkTrainer.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/NetworkTrainer.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/NetworkTrainer.cpp$(DependSuffix): NetworkTrainer.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/NetworkTrainer.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/NetworkTrainer.cpp$(DependSuffix) -MM NetworkTrainer.cpp

$(IntermediateDirectory)/NetworkTrainer.cpp$(PreprocessSuffix): NetworkTrainer.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/NetworkTrainer.cpp$(PreprocessSuffix)NetworkTrainer.cpp

$(IntermediateDirectory)/ShallowNetwork.cpp$(ObjectSuffix): ShallowNetwork.cpp $(IntermediateDirectory)/ShallowNetwork.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/robert/Documents/IsingNNParallel/ShallowNetwork.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/ShallowNetwork.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/ShallowNetwork.cpp$(DependSuffix): ShallowNetwork.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/ShallowNetwork.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/ShallowNetwork.cpp$(DependSuffix) -MM ShallowNetwork.cpp

$(IntermediateDirectory)/ShallowNetwork.cpp$(PreprocessSuffix): ShallowNetwork.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/ShallowNetwork.cpp$(PreprocessSuffix)ShallowNetwork.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Debug/


