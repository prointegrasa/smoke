Budowanie aplikacji:
1. Pobrać android sdk
2. Dodać ścieżkę do sdk w pliku local.properties
sdk.dir=
3. Zbudować za pomocą gradlew clean && gradlew build

Model można zaktualizować podmieniając pliki model_float16.tflite i squeeze.config w app/src/main/assets.