import sys
from PyQt5.QtWidgets import QApplication
from main_window import EnhancedTrafficManagementSystem

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = EnhancedTrafficManagementSystem()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc() 