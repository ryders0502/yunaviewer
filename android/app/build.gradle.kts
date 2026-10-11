plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "io.github.ryders0502.yunaviewer"
    // the Compose BOM needs 37 to compile; the app still targets 36
    compileSdk = 37

    defaultConfig {
        applicationId = "io.github.ryders0502.yunaviewer"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
    }

    buildTypes {
        // unsigned on purpose: build-apk.sh release signs it with the release key (kept outside the repo)
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2026.09.00"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("io.coil-kt.coil3:coil-compose:3.3.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20260814") // android.jar only has stubs of org.json
}
