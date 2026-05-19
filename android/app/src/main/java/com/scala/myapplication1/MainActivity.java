package com.scala.myapplication1;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.scala.myapplication1.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result;
    TextView cresult;
    ImageView imageView;
    Button picture;

    // Yeni modelin eğitim boyutuyla eşleşmeli (64x64)
    int imageSize = 64;

    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<String> permissionLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        cresult = findViewById(R.id.cresult);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        // Kamera sonucu için launcher
        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                activityResult -> {
                    if (activityResult.getResultCode() == RESULT_OK && activityResult.getData() != null) {
                        Bitmap image = (Bitmap) activityResult.getData().getExtras().get("data");
                        if (image != null) {
                            int dimension = Math.min(image.getWidth(), image.getHeight());
                            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                            imageView.setImageBitmap(image);
                            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                            classifyImage(image);
                        }
                    }
                }
        );

        // Kamera izni için launcher
        permissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(),
                isGranted -> {
                    if (isGranted) {
                        openCamera();
                    }
                }
        );

        picture.setOnClickListener(view -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                permissionLauncher.launch(Manifest.permission.CAMERA);
            }
        });
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraLauncher.launch(cameraIntent);
    }

    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Giriş boyutu 64x64x3 olarak güncellendi
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(
                    new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    // Model içinde Rescaling(1./255) katmanı var, ham piksel değerleri gönder
                    byteBuffer.putFloat((val >> 16) & 0xFF);
                    byteBuffer.putFloat((val >> 8) & 0xFF);
                    byteBuffer.putFloat(val & 0xFF);
                }
            }

            inputFeature0.loadBuffer(byteBuffer);
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            String[] classes = {"Beyti", "Domates Çorbası", "Mercimek Çorbası", "Sarma", "Su Böreği", "Tiramisu"};
            String[] descriptions = {"1 Porsiyon (400gr)", "1 Kase (244gr)", "1 Kase (248gr)", "1 Porsiyon (56gr)", "1 Porsiyon (120gr)", "1 Porsiyon (174gr)"};
            String[] calories = {"878 kcal", "102 kcal", "186 kcal", "92 kcal (Etsiz)\n99 kcal (Etli)", "300 kcal", "492 kcal"};

            // Softmax uygula (model from_logits=True ile eğitildi)
            float[] logits = outputFeature0.getFloatArray();
            float[] probs = softmax(logits);

            int maxPos = 0;
            for (int i = 1; i < classes.length; i++) {
                if (probs[i] > probs[maxPos]) {
                    maxPos = i;
                }
            }

            // Confidence threshold: %70 altında "Tanınamadı"
            if (probs[maxPos] < 0.7f) {
                result.setText("Tanınamadı");
                cresult.setText("Lütfen yemeği daha net çekin.");
            } else {
                result.setText(classes[maxPos]);
                cresult.setText(descriptions[maxPos] + "\n\n" + calories[maxPos]);
            }

            model.close();
        } catch (IOException e) {
            result.setText("Hata oluştu");
            cresult.setText(e.getMessage());
        }
    }

    private float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) if (v > max) max = v;

        float sum = 0;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }
}