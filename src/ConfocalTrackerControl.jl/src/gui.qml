import QtQuick 2.6
import QtQuick.Window 2.2
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.0
import org.julialang 1.0

ApplicationWindow {
	visible: true
	width: 1200
	height: 900
	title: qsTr("NIRTrack")
        onClosing: {
            Julia.window_close()
    }

	RowLayout {
		anchors.fill: parent
		JuliaCanvas {
			id: img_canvas
			paintFunction: cf_update
			Layout.alignment: Qt.AlignTop
			Layout.fillWidth: false
			Layout.fillHeight: false
			Layout.preferredWidth: 968
            Layout.preferredHeight: 732
		}
        
        RowLayout {
            spacing: 6
            Layout.alignment: Qt.AlignTop
            
            ColumnLayout {                
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Speed (mm/s)\navg 1s"
                    font.pointSize: 10
                }
                Text {
                    id: text_speed_avg
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 20
                }
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Recording"
                    font.pointSize: 10
                }
                Text {
                    id: text_recording_duration
	                Layout.alignment: Qt.AlignCenter
	                text: "N/A"
                    font.pointSize: 20
                }
            }
        }
        
        RowLayout {
            spacing: 6
            Layout.alignment: Qt.AlignTop
            
            ColumnLayout {
				Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Tracking"
                    font.pointSize: 10
                }
                Button {
                id: button_tracking
                Layout.alignment: Qt.AlignCenter
                text: "Start"
                font.pointSize: 10
                onClicked: { button_tracking.text = Julia.toggle_tracking().toString() }
            	}
            	Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Halt stage"
                    font.pointSize: 10
                }
				Button {
	                id: button_halt
	                Layout.alignment: Qt.AlignCenter
	                text: "Halt"
                    font.pointSize: 10
	                onClicked: {
	                Julia.send_halt_stage()
	                if (button_tracking.text == "Stop") {
	                    button_tracking.text = Julia.toggle_tracking().toString()
	               		}
	            	}
	        	}
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Recording"
                    font.pointSize: 10
                }
                Button {
                id: button_recording
                Layout.alignment: Qt.AlignCenter
                text: "Start"
                font.pointSize: 10
                onClicked: { button_recording.text = Julia.toggle_recording().toString() }
            	}
	        	Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Crosshair"
                    font.pointSize: 10
                }
				Button {
	                id: button_crosshair
	                Layout.alignment: Qt.AlignCenter
	                text: "Show"
                    font.pointSize: 10
	                onClicked: { button_crosshair.text = Julia.toggle_crosshair().toString() }
	        	}
	        	Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Ruler (100 um)"
                    font.pointSize: 10
                }
				Button {
	                id: button_ruler
	                Layout.alignment: Qt.AlignCenter
	                text: "Show"
                    font.pointSize: 10
	                onClicked: { button_ruler.text = Julia.toggle_ruler().toString() }
	        	}
                Text {
	                Layout.alignment: Qt.AlignCenter
	                text: "Deepnet output"
                    font.pointSize: 10
                }
				Button {
	                id: button_deepnetoutput
	                Layout.alignment: Qt.AlignCenter
	                text: "Hide"
                    font.pointSize: 10
	                onClicked: { button_deepnetoutput.text = Julia.toggle_deepnetoutput().toString() }
	        	}

			}
        }
	}
    
    JuliaSignals {
        signal updateCanvas()
        onUpdateCanvas: img_canvas.update()
        
        signal updateTextSpeedAvg(var str_speed)
        onUpdateTextSpeedAvg: text_speed_avg.text = str_speed
        
        signal updateTextRecordingDuration(var str_recording_duration)
        onUpdateTextRecordingDuration: text_recording_duration.text = str_recording_duration
    }
    
    
}
