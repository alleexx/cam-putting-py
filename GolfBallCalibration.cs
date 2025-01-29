using System;
using System.Drawing;
using System.Windows.Forms;
using AForge.Video;
using AForge.Video.DirectShow;
using IniParser;
using IniParser.Model;

namespace GolfBallCalibration
{
    public partial class CalibrationForm : Form
    {
        private FilterInfoCollection videoDevices;
        private VideoCaptureDevice videoSource;
        private IniData config;

        public CalibrationForm()
        {
            InitializeComponent();
            LoadWebcams();
            InitializeConfig();
        }

        private void LoadWebcams()
        {
            videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            foreach (FilterInfo device in videoDevices)
            {
                comboBoxWebcams.Items.Add(device.Name);
            }

            if (comboBoxWebcams.Items.Count > 0)
                comboBoxWebcams.SelectedIndex = 0;
        }

        private void InitializeConfig()
        {
            var parser = new FileIniDataParser();
            config = new IniData();

            if (System.IO.File.Exists("config.ini"))
            {
                config = parser.ReadFile("config.ini");
            }
            else
            {
                config["putting"]["hmin"] = "0";
                config["putting"]["smin"] = "0";
                config["putting"]["vmin"] = "0";
                config["putting"]["hmax"] = "255";
                config["putting"]["smax"] = "255";
                config["putting"]["vmax"] = "255";
            }

            // Load values into sliders
            trackBarHMin.Value = int.Parse(config["putting"]["hmin"]);
            trackBarSMin.Value = int.Parse(config["putting"]["smin"]);
            trackBarVMin.Value = int.Parse(config["putting"]["vmin"]);
            trackBarHMax.Value = int.Parse(config["putting"]["hmax"]);
            trackBarSMax.Value = int.Parse(config["putting"]["smax"]);
            trackBarVMax.Value = int.Parse(config["putting"]["vmax"]);
        }

        private void buttonStartPreview_Click(object sender, EventArgs e)
        {
            if (videoSource != null && videoSource.IsRunning)
                videoSource.Stop();

            videoSource = new VideoCaptureDevice(videoDevices[comboBoxWebcams.SelectedIndex].MonikerString);
            videoSource.NewFrame += VideoSource_NewFrame;
            videoSource.Start();
        }

        private void VideoSource_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            Bitmap frame = (Bitmap)eventArgs.Frame.Clone();
            pictureBoxPreview.Image = ApplyHSVFilter(frame);
        }

        private Bitmap ApplyHSVFilter(Bitmap frame)
        {
            // Apply HSV filtering logic here (pseudo-code):
            // 1. Convert to HSV
            // 2. Apply thresholds using trackBar values
            // 3. Return the filtered image
            return frame;
        }

        private void buttonSaveConfig_Click(object sender, EventArgs e)
        {
            config["putting"]["hmin"] = trackBarHMin.Value.ToString();
            config["putting"]["smin"] = trackBarSMin.Value.ToString();
            config["putting"]["vmin"] = trackBarVMin.Value.ToString();
            config["putting"]["hmax"] = trackBarHMax.Value.ToString();
            config["putting"]["smax"] = trackBarSMax.Value.ToString();
            config["putting"]["vmax"] = trackBarVMax.Value.ToString();

            var parser = new FileIniDataParser();
            parser.WriteFile("config.ini", config);
            MessageBox.Show("Configuration saved successfully!", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void CalibrationForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (videoSource != null && videoSource.IsRunning)
                videoSource.Stop();
        }
    }
}
