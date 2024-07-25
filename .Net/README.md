

![LponderGroup](logo/logo.png)

# Table Of Contents

- [Table Of Contents](#table-of-contents)
- [Documentation: Multi-Targeting Frameworks (.NET Framework 4.8 and .NET 8)](#documentation-multi-targeting-frameworks-net-framework-48-and-net-8)
  - [1. Create a Class Library with .NET Standard](#1-create-a-class-library-with-net-standard)
  - [2. Unload the Project](#2-unload-the-project)
  - [3. Modify the .csproj File](#3-modify-the-csproj-file)
  - [4. Reload the Project](#4-reload-the-project)
  - [5. Create a Sample Class](#5-create-a-sample-class)
  - [6. Build the Project](#6-build-the-project)
- [Supported Frameworks](#supported-frameworks)


# Documentation: Multi-Targeting Frameworks (.NET Framework 4.8 and .NET 8)

This documentation will guide you through the process of creating a class library that targets both .NET Framework 4.8 and .NET 8.

## 1. Create a Class Library with .NET Standard

1. Open Visual Studio.
2. Create a new project by navigating to **File > New > Project**.
3. Select **Class Library (.NET Standard)** and click **Next**.
4. Name the project `MultiTargetFrameworkLibrary` and click **Create**.

## 2. Unload the Project

To modify the project file, we need to unload it first:

1. In the **Solution Explorer**, right-click on the project (`MultiTargetFrameworkLibrary`).
2. Select **Unload Project**.

## 3. Modify the .csproj File

1. Right-click on the unloaded project and select **Edit MultiTargetFrameworkLibrary.csproj**.
2. Replace the content of the .csproj file with the following:

    ```xml
    <Project Sdk="Microsoft.NET.Sdk">
      <PropertyGroup>
        <TargetFrameworks>net48;net8.0</TargetFrameworks>
      </PropertyGroup>
    </Project>
    ```

    This configuration specifies that the project targets both .NET Framework 4.8 and .NET 8.

3. Save the changes and close the .csproj file.

## 4. Reload the Project

1. In the **Solution Explorer**, right-click on the unloaded project (`MultiTargetFrameworkLibrary`).
2. Select **Reload Project**.

## 5. Create a Sample Class

1. Open the `Class1.cs` file in the `HelloWorld` namespace.
2. Replace its content with the following code:

    ```csharp
    using System;

    namespace HelloWorld
    {
        public class Class1
        {
            public static string PrintFrameworkMessage()
            {
    #if NET48
                return "Hello World from .NET Framework 4.8";
    #elif NET8_0
                return "Hello World from .NET 8.0";
    #endif
            }
        }
    }
    ```

    This sample class demonstrates how to use conditional compilation symbols to differentiate between .NET Framework 4.8 and .NET 8.

## 6. Build the Project

1. In the **Solution Explorer**, right-click on the project (`MultiTargetFrameworkLibrary`).
2. Select **Build**.

# Supported Frameworks

If you want to target another framework, refer to the [Supported Frameworks documentation](https://learn.microsoft.com/en-us/dotnet/standard/frameworks#how-to-specify-target-frameworks).

Alternatively, see the table below for the supported Target Framework Monikers (TFMs):

| Target Framework               | TFM                |
| ------------------------------ | ------------------ |
| .NET 5+ (and .NET Core)        | netcoreapp1.0      |
|                                | netcoreapp1.1      |
|                                | netcoreapp2.0      |
|                                | netcoreapp2.1      |
|                                | netcoreapp2.2      |
|                                | netcoreapp3.0      |
|                                | netcoreapp3.1      |
|                                | net5.0*            |
|                                | net6.0*            |
|                                | net7.0*            |
|                                | net8.0*            |
| .NET Standard                  | netstandard1.0     |
|                                | netstandard1.1     |
|                                | netstandard1.2     |
|                                | netstandard1.3     |
|                                | netstandard1.4     |
|                                | netstandard1.5     |
|                                | netstandard1.6     |
|                                | netstandard2.0     |
|                                | netstandard2.1     |
| .NET Framework                 | net11              |
|                                | net20              |
|                                | net35              |
|                                | net40              |
|                                | net403             |
|                                | net45              |
|                                | net451             |
|                                | net452             |
|                                | net46              |
|                                | net461             |
|                                | net462             |
|                                | net47              |
|                                | net471             |
|                                | net472             |
|                                | net48              |
|                                | net481             |
| Windows Store                  | netcore [netcore45]|
|                                | netcore45 [win] [win8] |
|                                | netcore451 [win81] |
| .NET Micro Framework           | netmf              |
| Silverlight                    | sl4                |
|                                | sl5                |
| Windows Phone                  | wp [wp7]           |
|                                | wp7                |
|                                | wp75               |
|                                | wp8                |
|                                | wp81               |
|                                | wpa81              |
| Universal Windows Platform     | uap [uap10.0]      |
|                                | uap10.0 [win10] [netcore50] |



